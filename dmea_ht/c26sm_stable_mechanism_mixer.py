from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from dmea_ht.c17_residual import C17ResidualModel
from dmea_ht.mechanism_evidence_alignment import TEXT_MASK_KEYS


MECHANISM_NAMES = ("M1", "M2", "M3", "M4", "M5")
RELATION_NAMES = (
    "image_morphology",
    "text_morphology",
    "bio_immune",
    "bio_function",
    "text_opposition",
    "text_temporal",
)


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


def _masked_mean(nodes: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    weights = valid.to(nodes.dtype).unsqueeze(-1)
    return (nodes * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


class StableMechanismMixer(nn.Module):
    """Shared-message, fixed-topology, one-step five-mechanism mixer."""

    def __init__(self, hidden_dim: int, rho: float = 0.10) -> None:
        super().__init__()
        self.rho = float(rho)
        self.message_norm = nn.LayerNorm(hidden_dim)
        self.shared_message = nn.Linear(hidden_dim, hidden_dim)
        self.relation_logits = nn.ParameterDict(
            {name: nn.Parameter(torch.zeros(())) for name in RELATION_NAMES}
        )
        self.empty_mechanism_tokens = nn.Parameter(torch.zeros(len(MECHANISM_NAMES), hidden_dim))
        self.update_norm = nn.LayerNorm(hidden_dim)
        self.score_norm = nn.LayerNorm(hidden_dim)
        self.score_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.score_output = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.score_output.weight)
        nn.init.zeros_(self.score_output.bias)

    def _slot(
        self,
        sources: torch.Tensor,
        valid: torch.Tensor,
        relations: Sequence[str],
        slot_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        valid_float = valid.to(sources.dtype)
        source_mean = _masked_mean(sources, valid)
        messages = self.shared_message(self.message_norm(sources))
        gates = torch.stack([torch.sigmoid(self.relation_logits[name]) for name in relations])
        weights = valid_float * gates.unsqueeze(0)
        message = (messages * weights.unsqueeze(-1)).sum(dim=1) / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        available = valid.any(dim=1)
        empty = self.empty_mechanism_tokens[slot_index].unsqueeze(0).expand_as(message)
        message = torch.where(available.unsqueeze(-1), message, empty)
        source_mean = torch.where(available.unsqueeze(-1), source_mean, torch.zeros_like(source_mean))
        updated = self.update_norm(source_mean + self.rho * message)
        return updated, valid_float.sum(dim=1), ~available

    def forward(
        self,
        image_nodes: torch.Tensor,
        image_valid: torch.Tensor,
        text_nodes: torch.Tensor,
        text_valid: torch.Tensor,
        bio_nodes: torch.Tensor,
        bio_valid: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        image_source = _masked_mean(image_nodes, image_valid)
        image_available = image_valid.any(dim=1)
        text_morph_valid = torch.stack([text_valid[:, 0], text_valid[:, 3]], dim=1)
        text_morph_source = _masked_mean(torch.stack([text_nodes[:, 0], text_nodes[:, 3]], dim=1), text_morph_valid)
        text_morph_available = text_morph_valid.any(dim=1)

        specifications = (
            (
                torch.stack([image_source, text_morph_source], dim=1),
                torch.stack([image_available, text_morph_available], dim=1),
                ("image_morphology", "text_morphology"),
            ),
            (bio_nodes[:, 1:2], bio_valid[:, 1:2], ("bio_immune",)),
            (bio_nodes[:, 2:3], bio_valid[:, 2:3], ("bio_function",)),
            (text_nodes[:, 1:2], text_valid[:, 1:2], ("text_opposition",)),
            (text_nodes[:, 4:5], text_valid[:, 4:5], ("text_temporal",)),
        )
        nodes: List[torch.Tensor] = []
        counts: List[torch.Tensor] = []
        empty_masks: List[torch.Tensor] = []
        for index, (sources, valid, relations) in enumerate(specifications):
            node, count, empty = self._slot(sources, valid, relations, index)
            nodes.append(node)
            counts.append(count)
            empty_masks.append(empty)
        mechanism_nodes = torch.stack(nodes, dim=1)
        scores = self.score_output(torch.tanh(self.score_hidden(self.score_norm(mechanism_nodes)))).squeeze(-1)
        node_weights = torch.softmax(scores, dim=1)
        mechanism_core = torch.einsum("bk,bkh->bh", node_weights, mechanism_nodes)
        relation_gates = torch.stack([torch.sigmoid(self.relation_logits[name]) for name in RELATION_NAMES])
        return {
            "mechanism_nodes": mechanism_nodes,
            "node_weights": node_weights,
            "mechanism_core": mechanism_core,
            "valid_source_counts": torch.stack(counts, dim=1),
            "empty_slot_mask": torch.stack(empty_masks, dim=1),
            "relation_gates": relation_gates,
        }


class C26SMResidualMLP(nn.Module):
    """The single C17-style bounded prediction residual used by C26-SM."""

    def __init__(self, hidden_dim: int, dropout: float, residual_max: float) -> None:
        super().__init__()
        self.residual_max = float(residual_max)
        self.mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        output = self.mlp[-1]
        assert isinstance(output, nn.Linear)
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)

    def forward(self, mechanism_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.mlp(mechanism_state).squeeze(-1)
        return raw, self.residual_max * torch.tanh(raw)


class C26SMStableMechanismModel(nn.Module):
    """Frozen C17 pre-propagation extraction plus one stable single-model mixer."""

    def __init__(self, config: Dict[str, Any], seed: int) -> None:
        super().__init__()
        model_cfg = dict(config["model"])
        phase_cfg = dict(config["c26sm"])
        hidden_dim = int(model_cfg.get("hidden_dim", 256))
        self.frozen_c17 = C17ResidualModel(config, seed)
        checkpoint = Path(str(phase_cfg["c17_checkpoint"]).replace("{seed}", str(seed)))
        self.frozen_c17.load_state_dict(_checkpoint_state(checkpoint), strict=True)
        for parameter in self.frozen_c17.parameters():
            parameter.requires_grad = False
        self.frozen_c17.eval()

        self.mixer = StableMechanismMixer(hidden_dim, rho=float(phase_cfg["rho"]))
        self.residual_mlp = C26SMResidualMLP(
            hidden_dim,
            dropout=float(model_cfg.get("dropout", 0.15)),
            residual_max=float(phase_cfg["residual_max"]),
        )
        self.seed = int(seed)

    def train(self, mode: bool = True) -> "C26SMStableMechanismModel":
        super().train(mode)
        self.frozen_c17.eval()
        self.mixer.train(mode)
        self.residual_mlp.train(mode)
        return self

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        frozen = self.frozen_c17
        mea = frozen.mechanism_evidence_alignment
        with torch.no_grad():
            encoded = frozen.base_model.encode_modalities(batch)
            base_outputs = frozen.base_model.forward_from_encoded(batch, encoded)
            text_masks = {key: batch[key] for key in TEXT_MASK_KEYS}
            image = mea.image(encoded["image_tokens"], batch["image_mask"])
            text = mea.text(encoded["text_tokens"], batch["report_attention_mask"], text_masks)
            bio = mea.bio(encoded["bio_tokens"], batch["bio_missing_mask"])
            text_context = mea.mechanisms.relations["text_global"](text["nodes"][:, 5])
            text_context = text_context * text["valid"][:, 5].unsqueeze(-1).to(text_context.dtype)
            bio_context = mea.mechanisms.relations["bio_other"](bio["nodes"][:, 0])
            bio_context = bio_context * bio["valid"][:, 0].unsqueeze(-1).to(bio_context.dtype)
        mixed = self.mixer(
            image["nodes"].detach(), image["valid"], text["nodes"].detach(), text["valid"],
            bio["nodes"].detach(), bio["valid"],
        )
        context = text_context
        context = context + bio_context
        mechanism_state = mea.mechanisms.disease_norm(mixed["mechanism_core"] + context)
        raw_delta, delta = self.residual_mlp(mechanism_state)
        final_logit = base_outputs["logit"].detach() + delta
        node_norms = mixed["mechanism_nodes"].norm(dim=-1)
        return {
            "logit": final_logit,
            "prob": torch.sigmoid(final_logit),
            "base_logit": base_outputs["logit"].detach(),
            "base_prob": torch.sigmoid(base_outputs["logit"].detach()),
            "raw_delta": raw_delta,
            "delta_logit": delta,
            "mechanism_nodes": mixed["mechanism_nodes"],
            "mechanism_node_norms": node_norms,
            "mechanism_node_weights": mixed["node_weights"],
            "valid_source_counts": mixed["valid_source_counts"],
            "empty_slot_mask": mixed["empty_slot_mask"],
            "relation_gates": mixed["relation_gates"],
            "mechanism_core": mixed["mechanism_core"],
            "mechanism_state": mechanism_state,
            "mechanism_core_norm": mixed["mechanism_core"].norm(dim=-1),
            "context_norm": context.norm(dim=-1),
            "mechanism_final_norm": mechanism_state.norm(dim=-1),
        }


def c26sm_loss_terms(
    outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], loss_cfg: Mapping[str, Any]
) -> Dict[str, torch.Tensor]:
    labels = batch["label"]
    classification = F.binary_cross_entropy_with_logits(outputs["logit"], labels)
    residual = outputs["delta_logit"].square().mean()
    positive = labels > 0.5
    if bool(positive.any().item()):
        positive_preserve = F.relu(-outputs["delta_logit"][positive] - 0.05).mean()
    else:
        positive_preserve = outputs["delta_logit"].sum() * 0.0
    total = (
        classification
        + float(loss_cfg["lambda_residual"]) * residual
        + float(loss_cfg["lambda_positive_preserve"]) * positive_preserve
    )
    return {
        "total": total,
        "classification": classification,
        "residual": residual,
        "positive_preservation": positive_preserve,
    }


def propagation_capacity(model: C26SMStableMechanismModel) -> Dict[str, int | float]:
    mea = model.frozen_c17.mechanism_evidence_alignment
    original_modules = (mea.mechanisms, mea.role_scorer, mea.aggregator, mea.head)
    original = sum(parameter.numel() for module in original_modules for parameter in module.parameters())
    stable = sum(parameter.numel() for parameter in model.mixer.parameters())
    return {
        "original_c17_propagation_aggregation_parameters": int(original),
        "c26sm_stable_mixer_parameters": int(stable),
        "stable_over_original_ratio": float(stable / max(original, 1)),
    }
