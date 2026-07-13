from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from dmea_ht.c17_residual import C17ResidualModel
from dmea_ht.mechanism_evidence_alignment import TEXT_MASK_KEYS


STABLE_EVIDENCE_NODE_NAMES = tuple(
    [f"image_{name}" for name in ("diffuse", "texture", "structural", "nonspecific", "global")]
    + [f"text_{name}" for name in ("support", "opposition", "uncertainty", "nonspecific", "temporal", "global")]
    + [f"bio_{name}" for name in ("other_observed", "immune_observed", "function_observed")]
)


def _resolve_seed_path(value: str | Path, seed: int) -> Path:
    return Path(str(value).replace("{seed}", str(seed))).expanduser()


def _checkpoint_state(path: Path) -> Mapping[str, torch.Tensor]:
    if not path.exists():
        raise FileNotFoundError(f"C17 checkpoint does not exist: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    if not isinstance(state, Mapping):
        raise TypeError(f"Unsupported checkpoint payload at {path}")
    if any(str(key).startswith("module.") for key in state):
        return {str(key)[len("module.") :]: value for key, value in state.items()}
    return state


def stable_evidence_pool(nodes: torch.Tensor, valid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pool only real pre-propagation evidence nodes; valid counts are audit-only."""
    weights = valid.to(nodes.dtype).unsqueeze(-1)
    count = weights.sum(dim=1).squeeze(-1)
    pooled = (nodes * weights).sum(dim=1) / count.clamp_min(1.0).unsqueeze(-1)
    return pooled, count


class StableEvidencePoolingResidualHead(nn.Module):
    """C17-shaped bounded residual head over one frozen evidence-pool vector."""

    def __init__(self, hidden_dim: int, dropout: float, delta_max: float = 0.50) -> None:
        super().__init__()
        self.delta_max = float(delta_max)
        self.mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        output_layer = self.mlp[-1]
        assert isinstance(output_layer, nn.Linear)
        nn.init.zeros_(output_layer.weight)
        nn.init.zeros_(output_layer.bias)

    def forward(self, stable_evidence: torch.Tensor) -> Dict[str, torch.Tensor]:
        raw_delta = self.mlp(stable_evidence).squeeze(-1)
        delta_logit = self.delta_max * torch.tanh(raw_delta)
        return {"raw_delta": raw_delta, "delta_c22": delta_logit}


class C22StableEvidencePoolingModel(nn.Module):
    """Frozen C13/C17 pre-propagation evidence with one new residual head."""

    def __init__(self, config: Dict[str, Any], seed: int) -> None:
        super().__init__()
        model_cfg = dict(config.get("model", {}))
        c22_cfg = dict(config.get("c22", {}))
        hidden_dim = int(model_cfg.get("hidden_dim", 256))
        dropout = float(model_cfg.get("dropout", 0.15))

        reference_config = copy.deepcopy(config)
        reference_config["phase"] = "c17"
        reference_config.pop("c22", None)
        self.frozen_c17 = C17ResidualModel(reference_config, seed)
        c17_path = _resolve_seed_path(c22_cfg["c17_checkpoint"], seed)
        self.frozen_c17.load_state_dict(_checkpoint_state(c17_path), strict=True)
        for parameter in self.frozen_c17.parameters():
            parameter.requires_grad = False
        self.frozen_c17.eval()

        self.residual_head = StableEvidencePoolingResidualHead(
            hidden_dim=hidden_dim,
            dropout=dropout,
            delta_max=float(c22_cfg.get("delta_max", 0.50)),
        )
        self.seed = int(seed)

    def train(self, mode: bool = True) -> "C22StableEvidencePoolingModel":
        super().train(mode)
        self.frozen_c17.eval()
        self.residual_head.train(mode)
        return self

    @staticmethod
    def _stable_forward_inputs(
        reference: C17ResidualModel, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Mirror only the C13 encoder and the three real evidence projectors."""
        base = reference.base_model
        encoded = base.encode_modalities(batch)
        base_outputs = base.forward_from_encoded(batch, encoded)
        mea = reference.mechanism_evidence_alignment
        text_masks = {key: batch[key] for key in TEXT_MASK_KEYS}
        image = mea.image(encoded["image_tokens"], batch["image_mask"])
        text = mea.text(encoded["text_tokens"], batch["report_attention_mask"], text_masks)
        bio = mea.bio(encoded["bio_tokens"], batch["bio_missing_mask"])
        nodes = torch.cat([image["nodes"], text["nodes"], bio["nodes"]], dim=1)
        valid = torch.cat([image["valid"], text["valid"], bio["valid"]], dim=1)
        stable, count = stable_evidence_pool(nodes, valid)
        return base_outputs["logit"], stable, count

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            base_logit, stable_evidence, valid_evidence_count = self._stable_forward_inputs(
                self.frozen_c17, batch
            )
        residual = self.residual_head(stable_evidence)
        final_logit = base_logit + residual["delta_c22"]
        return {
            "logit": final_logit,
            "prob": torch.sigmoid(final_logit),
            "base_logit": base_logit,
            "base_prob": torch.sigmoid(base_logit),
            "raw_delta": residual["raw_delta"],
            "delta_c22": residual["delta_c22"],
            "stable_evidence_norm": stable_evidence.norm(dim=-1),
            "valid_evidence_count": valid_evidence_count,
        }


def c22_loss_terms(
    outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], loss_cfg: Mapping[str, Any]
) -> Dict[str, torch.Tensor]:
    """Return the fixed C22 objective and its auditable components."""
    criterion = F.binary_cross_entropy_with_logits(outputs["logit"], batch["label"], reduction="none")
    sample_weight = batch.get("sample_weight")
    if sample_weight is not None:
        cls_loss = (criterion * sample_weight).mean()
    else:
        cls_loss = criterion.mean()
    delta = outputs["delta_c22"]
    residual_loss = delta.square().mean()
    positive_mask = batch["label"] > 0.5
    allowed_negative_delta = float(loss_cfg.get("allowed_negative_delta", 0.05))
    if bool(positive_mask.any().item()):
        positive_preserve = F.relu(-delta[positive_mask] - allowed_negative_delta).mean()
    else:
        positive_preserve = delta.sum() * 0.0
    total = (
        cls_loss
        + float(loss_cfg.get("lambda_residual", 0.001)) * residual_loss
        + float(loss_cfg.get("lambda_positive_preserve", 0.02)) * positive_preserve
    )
    return {
        "total": total,
        "classification": cls_loss,
        "residual": residual_loss,
        "positive_preserve": positive_preserve,
    }
