from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict, Mapping

import torch
from torch import nn

from dmea_ht.mechanism_evidence_alignment import MechanismEvidenceAlignment, TEXT_MASK_KEYS
from dmea_ht.models import DMEAHTModel


def _resolve_seed_path(value: str | Path, seed: int) -> Path:
    return Path(str(value).replace("{seed}", str(seed))).expanduser()


def _checkpoint_state(path: Path) -> Mapping[str, torch.Tensor]:
    if not path.exists():
        raise FileNotFoundError(f"C17 checkpoint does not exist: {path}")
    payload = torch.load(path, map_location="cpu")
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    if not isinstance(state, Mapping):
        raise TypeError(f"Unsupported checkpoint payload at {path}")
    if any(str(key).startswith("module.") for key in state):
        return {str(key)[len("module.") :]: value for key, value in state.items()}
    return state


class MechanismResidualCorrectionHead(nn.Module):
    """Produce a bounded correction from DEMA mechanism evidence only."""

    def __init__(self, hidden_dim: int, dropout: float, delta_max: float = 0.50) -> None:
        super().__init__()
        self.delta_max = float(delta_max)
        input_dim = hidden_dim * 10 + 4
        self.mlp = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        output_layer = self.mlp[-1]
        assert isinstance(output_layer, nn.Linear)
        nn.init.zeros_(output_layer.weight)
        nn.init.zeros_(output_layer.bias)

    def forward(self, h_mechanism_correction: torch.Tensor) -> Dict[str, torch.Tensor]:
        raw_delta = self.mlp(h_mechanism_correction).squeeze(-1)
        delta_logit = self.delta_max * torch.tanh(raw_delta)
        return {"raw_delta": raw_delta, "delta_logit": delta_logit}


class C17ResidualModel(nn.Module):
    """Frozen C13 predictor plus a trainable DEMA mechanism residual."""

    def __init__(self, config: Dict, seed: int) -> None:
        super().__init__()
        model_cfg = dict(config.get("model", {}))
        c17_cfg = dict(config.get("c17", {}))
        hidden_dim = int(model_cfg.get("hidden_dim", 256))
        dropout = float(model_cfg.get("dropout", 0.15))
        num_heads = int(model_cfg.get("mea_num_heads", 4))

        base_config = copy.deepcopy(config)
        base_config["model"] = dict(model_cfg)
        base_config["model"]["use_mea"] = False
        self.base_model = DMEAHTModel(base_config)
        base_path = _resolve_seed_path(c17_cfg["base_checkpoint"], seed)
        self.base_model.load_state_dict(_checkpoint_state(base_path), strict=True)
        for parameter in self.base_model.parameters():
            parameter.requires_grad = False
        self.base_model.eval()

        self.mechanism_evidence_alignment = MechanismEvidenceAlignment(
            hidden_dim,
            dropout,
            num_heads=num_heads,
        )
        init_path_value = c17_cfg.get("init_mea_checkpoint")
        if init_path_value:
            init_path = _resolve_seed_path(init_path_value, seed)
            source_state = _checkpoint_state(init_path)
            prefix = "mechanism_evidence_alignment."
            mea_state = {
                str(key)[len(prefix) :]: value
                for key, value in source_state.items()
                if str(key).startswith(prefix)
            }
            if not mea_state:
                raise KeyError(f"No DEMA mechanism state found in C16 checkpoint: {init_path}")
            missing, unexpected = self.mechanism_evidence_alignment.load_state_dict(mea_state, strict=False)
            if missing or unexpected:
                raise RuntimeError(
                    f"C17 DEMA initialization mismatch: missing={list(missing)}, unexpected={list(unexpected)}"
                )

        self.residual_head = MechanismResidualCorrectionHead(
            hidden_dim,
            dropout,
            delta_max=float(c17_cfg.get("delta_max", 0.50)),
        )
        self.seed = int(seed)

    def train(self, mode: bool = True) -> "C17ResidualModel":
        super().train(mode)
        self.base_model.eval()
        return self

    @staticmethod
    def _mechanism_correction_features(mea_outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        mechanism_nodes = mea_outputs["mea_mechanism_nodes"]
        mechanism_valid = mea_outputs["mea_mechanism_valid"].to(mechanism_nodes.dtype)
        return torch.cat(
            [
                mechanism_nodes.flatten(start_dim=1),
                mea_outputs["mea_mechanism_state"],
                mea_outputs["mea_support_state"],
                mea_outputs["mea_opposition_state"],
                mea_outputs["mea_uncertainty_state"],
                mea_outputs["mea_conflict_state"],
                mechanism_valid[:, 1:5],
            ],
            dim=-1,
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            encoded = self.base_model.encode_modalities(batch)
            base_outputs = self.base_model.forward_from_encoded(batch, encoded)

        text_masks = {key: batch[key] for key in TEXT_MASK_KEYS}
        mea_outputs = self.mechanism_evidence_alignment(
            image_tokens=encoded["image_tokens"],
            image_mask=batch["image_mask"],
            text_tokens=encoded["text_tokens"],
            text_attention_mask=batch["report_attention_mask"],
            bio_tokens=encoded["bio_tokens"],
            bio_missing_mask=batch["bio_missing_mask"],
            text_masks=text_masks,
        )
        correction_features = self._mechanism_correction_features(mea_outputs)
        residual = self.residual_head(correction_features)
        base_logit = base_outputs["logit"]
        final_logit = base_logit + residual["delta_logit"]
        return {
            **mea_outputs,
            "mechanism_logit": mea_outputs["logit"],
            "logit": final_logit,
            "prob": torch.sigmoid(final_logit),
            "base_logit": base_logit,
            "base_prob": torch.sigmoid(base_logit),
            "raw_delta": residual["raw_delta"],
            "delta_logit": residual["delta_logit"],
            "mechanism_correction_norm": correction_features.norm(dim=-1),
        }
