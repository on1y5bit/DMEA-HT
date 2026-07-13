from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from dmea_ht.alignment import DSSAAlignment


class ImageEncoder(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(64, hidden_dim), nn.LayerNorm(hidden_dim))

    def forward(self, images: torch.Tensor, image_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, k, c, h, w = images.shape
        flat = images.view(batch * k, c, h, w)
        tokens = self.proj(self.backbone(flat)).view(batch, k, -1)
        mask = image_mask.unsqueeze(-1).float()
        masked_tokens = tokens * mask
        denom = mask.sum(dim=1).clamp_min(1.0)
        global_token = masked_tokens.sum(dim=1) / denom
        return tokens, global_token


class TextEncoder(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.proj = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.proj(self.embedding(input_ids))
        mask = attention_mask.unsqueeze(-1).float()
        global_token = (tokens * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return tokens, global_token


class BioEncoder(nn.Module):
    def __init__(self, bio_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.medical = nn.Sequential(
            nn.Linear(bio_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )
        self.observation = nn.Sequential(
            nn.Linear(bio_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )
        self.token_proj = nn.Linear(1, hidden_dim)

    def forward(
        self,
        bio_values: torch.Tensor,
        bio_missing_mask: torch.Tensor,
        bio_abnormal_flags: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        medical_input = torch.cat([bio_values, bio_abnormal_flags], dim=-1)
        bio_medical = self.medical(medical_input)
        bio_observation = self.observation(bio_missing_mask)
        bio_tokens = self.token_proj(bio_values.unsqueeze(-1))
        bio_global = bio_medical
        return bio_tokens, bio_global, bio_medical, bio_observation


class EvidenceRoleAlignment(nn.Module):
    roles = ["morphology", "immune", "function", "negative", "uncertain", "temporal"]

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.role_queries = nn.Parameter(torch.randn(len(self.roles), hidden_dim) * 0.02)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, image_tokens: torch.Tensor, text_tokens: torch.Tensor, bio_tokens: torch.Tensor) -> tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        batch = image_tokens.shape[0]
        source = torch.cat([image_tokens, text_tokens, bio_tokens], dim=1)
        queries = self.role_queries.unsqueeze(0).expand(batch, -1, -1)
        evidence_tokens, _ = self.attn(queries, source, source)
        raw_scores = self.score(evidence_tokens).squeeze(-1)
        evidence_scores = {role: raw_scores[:, i] for i, role in enumerate(self.roles)}
        role_loss = evidence_tokens.new_tensor(0.0)
        return evidence_tokens, evidence_scores, role_loss


class PatientAnchorFusion(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch = tokens.shape[0]
        query = self.anchor.expand(batch, -1, -1)
        fused, _ = self.attn(query, tokens, tokens)
        return self.norm(fused.squeeze(1))


class DiscordanceFusion(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )

    def pair(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([a, b, torch.abs(a - b), a * b], dim=-1))

    def forward(self, image_global: torch.Tensor, text_global: torch.Tensor, bio_global: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "d_img_txt": self.pair(image_global, text_global),
            "d_img_bio": self.pair(image_global, bio_global),
            "d_txt_bio": self.pair(text_global, bio_global),
        }


class EvidenceConservationClassifier(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.e_img = nn.Linear(hidden_dim, 1)
        self.e_text = nn.Linear(hidden_dim, 1)
        self.e_bio = nn.Linear(hidden_dim, 1)
        self.e_synergy = nn.Linear(hidden_dim, 1)
        self.e_negative = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        image_global: torch.Tensor,
        text_global: torch.Tensor,
        bio_medical: torch.Tensor,
        z_patient: torch.Tensor,
        negative_token: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        e_img = self.e_img(image_global).squeeze(-1)
        e_text = self.e_text(text_global).squeeze(-1)
        e_bio = self.e_bio(bio_medical).squeeze(-1)
        e_synergy = self.e_synergy(z_patient).squeeze(-1)
        e_negative = torch.relu(self.e_negative(negative_token).squeeze(-1))
        logit = e_img + e_text + e_bio + e_synergy - e_negative
        return {
            "logit": logit,
            "prob": torch.sigmoid(logit),
            "e_img": e_img,
            "e_text": e_text,
            "e_bio": e_bio,
            "e_synergy": e_synergy,
            "e_negative": e_negative,
        }


class TextEvidenceAnchor(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float, num_heads: int = 4) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, batch_first=True)
        self.anchor_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.morphology_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        text_feature: torch.Tensor,
        text_tokens: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        if text_tokens is not None:
            batch = text_tokens.shape[0]
            query = self.query.expand(batch, -1, -1)
            key_padding_mask = attention_mask == 0 if attention_mask is not None else None
            attended, _ = self.attn(query, text_tokens, text_tokens, key_padding_mask=key_padding_mask)
            source = attended.squeeze(1)
        else:
            source = text_feature
        anchor = self.anchor_proj(source)
        logit = self.morphology_head(anchor).squeeze(-1)
        return {
            "text_morphology_anchor": anchor,
            "text_morphology_logit": logit,
            "text_morphology_prob": torch.sigmoid(logit),
        }


class DMEAHTModel(nn.Module):
    def __init__(self, config: Dict) -> None:
        super().__init__()
        model_cfg = config.get("model", {})
        hidden_dim = int(model_cfg.get("hidden_dim", 256))
        dropout = float(model_cfg.get("dropout", 0.15))
        self.variant = str(model_cfg.get("variant", "dmea"))
        self.image_encoder = ImageEncoder(hidden_dim, dropout)
        self.text_encoder = TextEncoder(int(model_cfg.get("text_vocab_size", 50000)), hidden_dim, dropout)
        self.bio_encoder = BioEncoder(int(model_cfg.get("bio_dim", 32)), hidden_dim, dropout)
        self.evidence = EvidenceRoleAlignment(hidden_dim)
        self.anchor = PatientAnchorFusion(hidden_dim)
        self.discordance = DiscordanceFusion(hidden_dim, dropout)
        self.classifier = EvidenceConservationClassifier(hidden_dim)
        self.use_text_morphology_head = bool(model_cfg.get("use_text_morphology_head", False))
        self.use_image_morphology_head = bool(model_cfg.get("use_image_morphology_head", False))
        self.use_text_evidence_anchor = bool(model_cfg.get("use_text_evidence_anchor", False))
        self.fuse_text_morphology_anchor = bool(model_cfg.get("fuse_text_morphology_anchor", False))
        self.text_evidence_anchor = (
            TextEvidenceAnchor(hidden_dim, dropout) if self.use_text_evidence_anchor and self.use_text_morphology_head else None
        )
        self.text_morphology_head = (
            nn.Linear(hidden_dim, 1) if self.use_text_morphology_head and self.text_evidence_anchor is None else None
        )
        self.image_morphology_head = nn.Linear(hidden_dim, 1) if self.use_image_morphology_head else None
        self.use_dssa = bool(model_cfg.get("use_dssa", False))
        self.dssa = (
            DSSAAlignment(
                hidden_dim=hidden_dim,
                dropout=dropout,
                shared_dim=int(model_cfg.get("dssa_shared_dim", hidden_dim)),
                temperature=float(model_cfg.get("dssa_temperature", 0.10)),
                anchor_temperature=float(model_cfg.get("dssa_anchor_temperature", 0.10)),
                margin_proto=float(model_cfg.get("dssa_margin_proto", 0.0)),
                specific_variance_gamma=float(model_cfg.get("dssa_specific_variance_gamma", 0.50)),
                residual_scale=float(model_cfg.get("dssa_residual_scale", 0.10)),
            )
            if self.use_dssa
            else None
        )
        self.baseline_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def _auxiliary_outputs(
        self,
        image_global: torch.Tensor,
        text_global: torch.Tensor,
        text_tokens: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        outputs: Dict[str, torch.Tensor] = {}
        if self.text_evidence_anchor is not None:
            outputs.update(self.text_evidence_anchor(text_global, text_tokens, attention_mask))
        if self.text_morphology_head is not None:
            logit = self.text_morphology_head(text_global).squeeze(-1)
            outputs["text_morphology_logit"] = logit
            outputs["text_morphology_prob"] = torch.sigmoid(logit)
        if self.image_morphology_head is not None:
            outputs["image_morphology_logit"] = self.image_morphology_head(image_global).squeeze(-1)
        return outputs

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        image_tokens, image_global = self.image_encoder(batch["images"], batch["image_mask"])
        text_tokens, text_global = self.text_encoder(batch["report_input_ids"], batch["report_attention_mask"])
        bio_tokens, bio_global, bio_medical, _bio_observation = self.bio_encoder(
            batch["bio_values"], batch["bio_missing_mask"], batch["bio_abnormal_flags"]
        )
        aux_outputs = self._auxiliary_outputs(image_global, text_global, text_tokens, batch["report_attention_mask"])

        if self.dssa is not None:
            outputs = self.dssa(image_global, text_global, bio_global, batch)
            outputs.update(aux_outputs)
            return outputs

        if self.variant in {"image_only", "text_only", "bio_only", "concat"}:
            parts = {
                "image_only": [image_global, torch.zeros_like(text_global), torch.zeros_like(bio_global)],
                "text_only": [torch.zeros_like(image_global), text_global, torch.zeros_like(bio_global)],
                "bio_only": [torch.zeros_like(image_global), torch.zeros_like(text_global), bio_global],
                "concat": [image_global, text_global, bio_global],
            }[self.variant]
            logit = self.baseline_head(torch.cat(parts, dim=-1)).squeeze(-1)
            return {"logit": logit, "prob": torch.sigmoid(logit), **aux_outputs}

        evidence_tokens, evidence_scores, role_loss = self.evidence(image_tokens, text_tokens, bio_tokens)
        token_parts = [image_tokens, text_tokens, bio_tokens, evidence_tokens]
        if self.fuse_text_morphology_anchor and "text_morphology_anchor" in aux_outputs:
            token_parts.append(aux_outputs["text_morphology_anchor"].unsqueeze(1))
        tokens = torch.cat(token_parts, dim=1)
        z_patient = self.anchor(tokens)
        discordance = self.discordance(image_global, text_global, bio_global)
        negative_token = evidence_tokens[:, self.evidence.roles.index("negative"), :]
        outputs = self.classifier(image_global, text_global, bio_medical, z_patient, negative_token)
        outputs.update(aux_outputs)
        outputs["role_alignment_loss"] = role_loss
        for key, value in evidence_scores.items():
            outputs[f"evidence_{key}"] = value
        for key, value in discordance.items():
            outputs[key] = value.norm(dim=-1)
        return outputs
