from __future__ import annotations

from typing import Any, Dict

import torch
from torch import nn

from dmea_ht.c47_drfe import C47DRFEModel


HEAD_PREFIXES = (
    "stream_adapters.",
    "trajectory_encoder.",
    "patient_readout.",
    "classifier.",
)


class LowRankAdapter(nn.Module):
    """Identity-initialized bottleneck transport for one frozen evidence stream."""

    def __init__(self, hidden_dim: int, rank: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.down = nn.Linear(hidden_dim, rank, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.up = nn.Linear(rank, hidden_dim, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        correction = self.up(self.dropout(torch.nn.functional.gelu(self.down(self.norm(states)))))
        return states + 0.5 * correction


class C54LRRAModel(C47DRFEModel):
    """Adapt frozen raw/aligned evidence through six low-rank stream transports."""

    def __init__(self, config: Dict[str, Any], seed: int) -> None:
        source_config = dict(config)
        source_config["c47"] = dict(config["c54"])
        super().__init__(source_config, seed)
        for name in ("stream_encoder", "patient_readout", "classifier"):
            delattr(self, name)
        model_cfg = dict(config["model"])
        hidden_dim = self.hidden_dim
        dropout = float(model_cfg["dropout"])
        rank = int(config["c54"]["adapter_rank"])
        self.stream_adapters = nn.ModuleList(
            [LowRankAdapter(hidden_dim, rank, dropout) for _ in range(6)]
        )
        self.trajectory_encoder = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )
        self.patient_readout = nn.Sequential(
            nn.LayerNorm(hidden_dim * 8),
            nn.Linear(hidden_dim * 8, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        expected_order = [
            "raw_image",
            "raw_text",
            "raw_bio",
            "aligned_image",
            "aligned_text",
            "aligned_bio",
        ]
        if list(config["c54"]["stream_order"]) != expected_order:
            raise RuntimeError("C54 stream order is fixed to raw then aligned image/text/bio")

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            source = self._source_states(batch)
        adapted_states = torch.stack(
            [adapter(source["states"][:, :, index]) for index, adapter in enumerate(self.stream_adapters)],
            dim=2,
        )
        summary, available, latest_mask, history_mask, latest_weights, history_weights = self._fixed_trajectory_statistics(
            adapted_states, source["valid"], source["visit_mask"]
        )
        stream_tokens = self.trajectory_encoder(summary)
        safe_available = available.clone()
        no_evidence = ~safe_available.any(dim=1)
        if bool(no_evidence.any().item()):
            safe_available[no_evidence, 0] = True
        weights = safe_available.to(stream_tokens.dtype)
        stream_tokens = stream_tokens * weights.unsqueeze(-1)
        denominator = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        consensus = (stream_tokens * weights.unsqueeze(-1)).sum(dim=1) / denominator
        discordance = (
            (stream_tokens - consensus.unsqueeze(1)).abs() * weights.unsqueeze(-1)
        ).sum(dim=1) / denominator
        patient_state = self.patient_readout(
            torch.cat([stream_tokens.flatten(start_dim=1), consensus, discordance], dim=-1)
        )
        logit = self.classifier(patient_state).squeeze(-1)
        evidence_tokens = torch.stack(
            [consensus, discordance, consensus * discordance, stream_tokens.mean(dim=1)], dim=1
        )
        evidence_valid = safe_available.any(dim=1).unsqueeze(1).expand(-1, 4)
        attention = evidence_valid.to(stream_tokens.dtype)
        attention = attention / attention.sum(dim=1, keepdim=True).clamp_min(1.0)
        bio_state = adapted_states[:, :, 5]
        bio_valid = source["valid"][:, :, 5]
        bio_weights = bio_valid.to(bio_state.dtype)
        bio_denominator = bio_weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        bio_state = (bio_state * bio_weights.unsqueeze(-1)).sum(dim=1) / bio_denominator
        return {
            "logit": logit,
            "prob": torch.sigmoid(logit),
            "patient_state": patient_state,
            "attended_evidence": consensus,
            "bio_state": bio_state,
            "evidence_tokens": evidence_tokens,
            "evidence_valid": evidence_valid,
            "latest_bio_valid": (latest_mask.unsqueeze(-1) & (~batch["bio_missing_mask"].bool())).any(dim=1),
            "history_bio_valid": (history_mask.unsqueeze(-1) & (~batch["bio_missing_mask"].bool())).any(dim=1),
            "latest_weights": latest_weights,
            "history_weights": history_weights,
            "latest_mask": latest_mask,
            "history_mask": history_mask,
            "attention": attention,
            "trajectory_available": available,
            "stream_tokens": stream_tokens,
            "stream_valid": safe_available,
            "adapted_states": adapted_states,
            "consensus_state": consensus,
            "discordance_state": discordance,
        }


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def trainable_parameter_names(model: nn.Module) -> list[str]:
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]
