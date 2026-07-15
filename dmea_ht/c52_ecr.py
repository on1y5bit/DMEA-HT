from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn.functional as F
from torch import nn

from dmea_ht.c47_drfe import C47DRFEModel


HEAD_PREFIXES = (
    "stream_encoder.",
    "patient_query",
    "patient_attention.",
    "patient_norm.",
    "evidence_heads.",
    "synergy_head.",
    "opposition_head.",
)


class C52ECRModel(C47DRFEModel):
    """Conserve separate modality evidence around one patient-level query."""

    def __init__(self, config: Dict[str, Any], seed: int) -> None:
        source_config = dict(config)
        source_config["c47"] = dict(config["c52"])
        super().__init__(source_config, seed)
        for name in ("stream_encoder", "patient_readout", "classifier"):
            delattr(self, name)
        model_cfg = dict(config["model"])
        hidden_dim = self.hidden_dim
        dropout = float(model_cfg["dropout"])
        self.stream_encoder = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )
        self.patient_query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.patient_attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads=int(model_cfg["mea_num_heads"]),
            dropout=dropout,
            batch_first=True,
        )
        self.patient_norm = nn.LayerNorm(hidden_dim)
        self.evidence_heads = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(3)])
        self.synergy_head = nn.Linear(hidden_dim, 1)
        self.opposition_head = nn.Linear(hidden_dim, 1)
        expected_order = [
            "raw_image",
            "raw_text",
            "raw_bio",
            "aligned_image",
            "aligned_text",
            "aligned_bio",
        ]
        if list(config["c52"]["stream_order"]) != expected_order:
            raise RuntimeError("C52 stream order is fixed to raw then aligned image/text/bio")

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            source = self._source_states(batch)
        summary, available, latest_mask, history_mask, latest_weights, history_weights = self._fixed_trajectory_statistics(
            source["states"], source["valid"], source["visit_mask"]
        )
        stream_tokens = self.stream_encoder(summary)
        safe_available = available.clone()
        no_evidence = ~safe_available.any(dim=1)
        if bool(no_evidence.any().item()):
            safe_available[no_evidence, 0] = True
        query = self.patient_query.expand(stream_tokens.shape[0], -1, -1)
        attended, attention_weights = self.patient_attention(
            query,
            stream_tokens,
            stream_tokens,
            key_padding_mask=~safe_available,
            need_weights=True,
        )
        patient_state = self.patient_norm(attended.squeeze(1) + query.squeeze(1))
        weights = safe_available.to(stream_tokens.dtype)
        modality_states = []
        for modality_index in range(3):
            pair = torch.stack(
                [stream_tokens[:, modality_index], stream_tokens[:, modality_index + 3]], dim=1
            )
            pair_weights = weights[:, (modality_index, modality_index + 3)]
            denominator = pair_weights.sum(dim=1, keepdim=True).clamp_min(1.0)
            modality_states.append(
                (pair * pair_weights.unsqueeze(-1)).sum(dim=1) / denominator
            )
        modality_states_tensor = torch.stack(modality_states, dim=1)
        evidence_scores = torch.cat(
            [head(modality_states_tensor[:, index]) for index, head in enumerate(self.evidence_heads)],
            dim=-1,
        )
        consensus = (modality_states_tensor * weights[:, :3].unsqueeze(-1)).sum(dim=1) / weights[:, :3].sum(
            dim=1, keepdim=True
        ).clamp_min(1.0)
        modality_discordance = (
            (modality_states_tensor - consensus.unsqueeze(1)).abs() * weights[:, :3].unsqueeze(-1)
        ).sum(dim=1) / weights[:, :3].sum(dim=1, keepdim=True).clamp_min(1.0)
        synergy = self.synergy_head(patient_state).squeeze(-1)
        opposition = F.relu(self.opposition_head(modality_discordance).squeeze(-1))
        logit = evidence_scores.sum(dim=-1) + synergy - opposition
        attention = attention_weights.squeeze(1)
        evidence_tokens = torch.stack([patient_state, consensus, modality_discordance, modality_discordance], dim=1)
        evidence_valid = safe_available.any(dim=1).unsqueeze(1).expand(-1, 4)
        bio_state = modality_states_tensor[:, 2]
        return {
            "logit": logit,
            "prob": torch.sigmoid(logit),
            "patient_state": patient_state,
            "attended_evidence": patient_state,
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
            "modality_states": modality_states_tensor,
            "evidence_scores": evidence_scores,
            "synergy": synergy,
            "opposition": opposition,
            "consensus_state": consensus,
            "discordance_state": modality_discordance,
        }


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def trainable_parameter_names(model: nn.Module) -> list[str]:
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]
