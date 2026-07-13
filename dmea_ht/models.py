from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict, Mapping

import torch
import torch.nn.functional as F
from torch import nn

from dmea_ht.mechanism_evidence_alignment import MechanismEvidenceAlignment, TEXT_MASK_KEYS


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
        self.baseline_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.use_mea = bool(model_cfg.get("use_mea", False))
        self.mechanism_evidence_alignment = (
            MechanismEvidenceAlignment(
                hidden_dim,
                dropout,
                num_heads=int(model_cfg.get("mea_num_heads", 4)),
            )
            if self.use_mea
            else None
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

    def encode_modalities(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        image_tokens, image_global = self.image_encoder(batch["images"], batch["image_mask"])
        text_tokens, text_global = self.text_encoder(batch["report_input_ids"], batch["report_attention_mask"])
        bio_tokens, bio_global, bio_medical, _bio_observation = self.bio_encoder(
            batch["bio_values"], batch["bio_missing_mask"], batch["bio_abnormal_flags"]
        )
        return {
            "image_tokens": image_tokens,
            "image_global": image_global,
            "text_tokens": text_tokens,
            "text_global": text_global,
            "bio_tokens": bio_tokens,
            "bio_global": bio_global,
            "bio_medical": bio_medical,
        }

    def forward_from_encoded(
        self,
        batch: Dict[str, torch.Tensor],
        encoded: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        image_tokens = encoded["image_tokens"]
        image_global = encoded["image_global"]
        text_tokens = encoded["text_tokens"]
        text_global = encoded["text_global"]
        bio_tokens = encoded["bio_tokens"]
        bio_global = encoded["bio_global"]
        bio_medical = encoded["bio_medical"]
        aux_outputs = self._auxiliary_outputs(image_global, text_global, text_tokens, batch["report_attention_mask"])

        if self.mechanism_evidence_alignment is not None:
            text_masks = {key: batch[key] for key in TEXT_MASK_KEYS}
            outputs = self.mechanism_evidence_alignment(
                image_tokens=image_tokens,
                image_mask=batch["image_mask"],
                text_tokens=text_tokens,
                text_attention_mask=batch["report_attention_mask"],
                bio_tokens=bio_tokens,
                bio_missing_mask=batch["bio_missing_mask"],
                text_masks=text_masks,
            )
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

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.forward_from_encoded(batch, self.encode_modalities(batch))


def _resolve_seed_path(value: str | Path, seed: int) -> Path:
    return Path(str(value).replace("{seed}", str(seed))).expanduser()


def _checkpoint_state(path: Path) -> Mapping[str, torch.Tensor]:
    if not path.exists():
        raise FileNotFoundError(f"checkpoint does not exist: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    if not isinstance(state, Mapping):
        raise TypeError(f"unsupported checkpoint payload at {path}")
    if any(str(key).startswith("module.") for key in state):
        return {str(key)[len("module.") :]: value for key, value in state.items()}
    return state


class _DirectionalScalarMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        output_layer = self.net[-1]
        assert isinstance(output_layer, nn.Linear)
        nn.init.zeros_(output_layer.weight)
        nn.init.zeros_(output_layer.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


class MechanismSupportResidualHead(_DirectionalScalarMLP):
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return 0.50 * torch.sigmoid(super().forward(features))


class MechanismOppositionResidualHead(_DirectionalScalarMLP):
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return 0.50 * torch.sigmoid(super().forward(features))


class MechanismSupportGate(_DirectionalScalarMLP):
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(super().forward(features))


class MechanismOppositionGate(_DirectionalScalarMLP):
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(super().forward(features))


class C18DirectionalResidualModel(nn.Module):
    """Frozen C13 logit plus bounded, evidence-directional residuals."""

    def __init__(self, config: Dict, seed: int) -> None:
        super().__init__()
        model_cfg = dict(config.get("model", {}))
        c18_cfg = dict(config.get("c18", {}))
        hidden_dim = int(model_cfg.get("hidden_dim", 256))
        dropout = float(model_cfg.get("dropout", 0.15))
        num_heads = int(model_cfg.get("mea_num_heads", 4))

        base_config = copy.deepcopy(config)
        base_config["model"] = dict(model_cfg)
        base_config["model"]["use_mea"] = False
        self.base_model = DMEAHTModel(base_config)
        base_path = _resolve_seed_path(c18_cfg["base_checkpoint"], seed)
        self.base_model.load_state_dict(_checkpoint_state(base_path), strict=True)
        for parameter in self.base_model.parameters():
            parameter.requires_grad = False
        self.base_model.eval()

        self.mechanism_evidence_alignment = MechanismEvidenceAlignment(
            hidden_dim,
            dropout,
            num_heads=num_heads,
        )
        init_path_value = c18_cfg.get("init_mea_checkpoint")
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
                raise KeyError(f"no DEMA mechanism state found in checkpoint: {init_path}")
            missing, unexpected = self.mechanism_evidence_alignment.load_state_dict(mea_state, strict=False)
            if missing or unexpected:
                raise RuntimeError(
                    f"C18 DEMA initialization mismatch: missing={list(missing)}, unexpected={list(unexpected)}"
                )

        self.support_head = MechanismSupportResidualHead(hidden_dim * 4 + 8, hidden_dim, dropout)
        self.opposition_head = MechanismOppositionResidualHead(hidden_dim * 4 + 8, hidden_dim, dropout)
        self.support_gate = MechanismSupportGate(hidden_dim * 3 + 8, hidden_dim, dropout)
        self.opposition_gate = MechanismOppositionGate(hidden_dim * 3 + 8, hidden_dim, dropout)
        self.seed = int(seed)

    def train(self, mode: bool = True) -> "C18DirectionalResidualModel":
        super().train(mode)
        self.base_model.eval()
        return self

    @staticmethod
    def _directional_features(mea_outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        mechanism_state = mea_outputs["mea_mechanism_state"]
        support_state = mea_outputs["mea_support_state"]
        opposition_state = mea_outputs["mea_opposition_state"]
        uncertainty_state = mea_outputs["mea_uncertainty_state"]
        conflict_state = mea_outputs["mea_conflict_state"]
        strengths = mea_outputs["mea_strengths"]
        mechanism_valid = mea_outputs["mea_mechanism_valid"].to(mechanism_state.dtype)
        support_features = torch.cat(
            [support_state, mechanism_state, uncertainty_state, conflict_state, strengths, mechanism_valid],
            dim=-1,
        )
        opposition_features = torch.cat(
            [opposition_state, mechanism_state, uncertainty_state, conflict_state, strengths, mechanism_valid],
            dim=-1,
        )
        support_gate_features = torch.cat(
            [support_state, opposition_state, conflict_state, strengths, mechanism_valid],
            dim=-1,
        )
        opposition_gate_features = torch.cat(
            [opposition_state, support_state, conflict_state, strengths, mechanism_valid],
            dim=-1,
        )
        return {
            "support": support_features,
            "opposition": opposition_features,
            "support_gate": support_gate_features,
            "opposition_gate": opposition_gate_features,
        }

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
        features = self._directional_features(mea_outputs)
        support_delta = self.support_head(features["support"])
        opposition_delta = self.opposition_head(features["opposition"])
        support_gate = self.support_gate(features["support_gate"])
        opposition_gate = self.opposition_gate(features["opposition_gate"])
        conflict_suppression = mea_outputs["conflict_suppression"]
        effective_support_delta = support_gate * conflict_suppression * support_delta
        effective_opposition_delta = opposition_gate * conflict_suppression * opposition_delta
        directional_delta = effective_support_delta - effective_opposition_delta
        base_logit = base_outputs["logit"]
        final_logit = base_logit + directional_delta
        return {
            **mea_outputs,
            "mechanism_logit": mea_outputs["logit"],
            "logit": final_logit,
            "prob": torch.sigmoid(final_logit),
            "base_logit": base_logit,
            "base_prob": torch.sigmoid(base_logit),
            "support_delta": support_delta,
            "opposition_delta": opposition_delta,
            "support_gate": support_gate,
            "opposition_gate": opposition_gate,
            "effective_support_delta": effective_support_delta,
            "effective_opposition_delta": effective_opposition_delta,
            "directional_delta": directional_delta,
            "delta_logit": directional_delta,
            "directional_feature_norm": features["support"].norm(dim=-1),
        }


class MonotonicSupportCalibrator(nn.Module):
    """Fixed positive-slope calibration over frozen support evidence."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("raw_a_support", torch.tensor(0.54132485))
        self.register_buffer("b_support", torch.tensor(0.0))

    @property
    def a_support(self) -> torch.Tensor:
        return F.softplus(self.raw_a_support)

    def forward(self, normalized_support_strength: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.a_support * normalized_support_strength + self.b_support)


class MonotonicOppositionCalibrator(nn.Module):
    """Fixed positive-slope calibration over frozen opposition evidence."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("raw_a_opposition", torch.tensor(0.54132485))
        self.register_buffer("b_opposition", torch.tensor(0.0))

    @property
    def a_opposition(self) -> torch.Tensor:
        return F.softplus(self.raw_a_opposition)

    def forward(self, normalized_opposition_strength: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.a_opposition * normalized_opposition_strength + self.b_opposition)


class EvidenceMagnitudeHead(nn.Module):
    """Learn only a bounded correction magnitude from polarity confidence."""

    def __init__(self, hidden_dim: int, dropout: float, magnitude_max: float = 0.20) -> None:
        super().__init__()
        self.magnitude_max = float(magnitude_max)
        self.net = nn.Sequential(
            nn.LayerNorm(4),
            nn.Linear(4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        output_layer = self.net[-1]
        assert isinstance(output_layer, nn.Linear)
        nn.init.zeros_(output_layer.weight)
        nn.init.zeros_(output_layer.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.magnitude_max * torch.sigmoid(self.net(features).squeeze(-1))


class C19PolarityLockedResidualModel(nn.Module):
    """Frozen C17 evidence with polarity-locked, magnitude-only residual correction."""

    def __init__(self, config: Dict, seed: int) -> None:
        super().__init__()
        model_cfg = dict(config.get("model", {}))
        c19_cfg = dict(config.get("c19", {}))
        hidden_dim = int(model_cfg.get("hidden_dim", 256))
        dropout = float(model_cfg.get("dropout", 0.15))
        c17_checkpoint = _resolve_seed_path(c19_cfg["c17_checkpoint"], seed)
        c13_checkpoint = c19_cfg["c13_checkpoint"]

        # Reconstruct the C17 module only to load its full validation-selected state.
        # Every C17 parameter is frozen before the new magnitude head is exposed.
        from dmea_ht.c17_residual import C17ResidualModel

        c17_config = copy.deepcopy(config)
        c17_config["phase"] = "c17"
        c17_config.pop("c19", None)
        c17_config["c17"] = {
            "variant": "positive_preserve",
            "base_checkpoint": c13_checkpoint,
            "init_mea_checkpoint": str(c17_checkpoint),
            "delta_max": 0.50,
        }
        self.frozen_c17 = C17ResidualModel(c17_config, seed)
        self.frozen_c17.load_state_dict(_checkpoint_state(c17_checkpoint), strict=True)
        for parameter in self.frozen_c17.parameters():
            parameter.requires_grad = False
        self.frozen_c17.eval()

        self.support_calibrator = MonotonicSupportCalibrator()
        self.opposition_calibrator = MonotonicOppositionCalibrator()
        self.magnitude_head = EvidenceMagnitudeHead(
            hidden_dim,
            dropout,
            magnitude_max=float(c19_cfg.get("magnitude_max", 0.20)),
        )
        # A tiny positive scale preserves exact C17 behavior at initialization while
        # retaining a direct, finite gradient for the learnable magnitude path.
        self.residual_scale = nn.Parameter(
            torch.tensor(float(c19_cfg.get("residual_scale_init", 1e-9)), dtype=torch.float32)
        )
        self.polarity_temperature = float(c19_cfg.get("polarity_temperature", 1.0))

    def train(self, mode: bool = True) -> "C19PolarityLockedResidualModel":
        super().train(mode)
        self.frozen_c17.eval()
        return self

    @staticmethod
    def _normalize_evidence(frozen_outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        support = torch.sigmoid(frozen_outputs["evidence_support_strength"])
        opposition = torch.sigmoid(frozen_outputs["evidence_opposition_strength"])
        uncertainty = frozen_outputs["evidence_uncertainty_strength"].clamp(0.0, 1.0)
        conflict = frozen_outputs["evidence_conflict_score"].clamp(0.0, 1.0)
        temporal_conflict = frozen_outputs["evidence_temporal_conflict_score"].clamp(0.0, 1.0)
        morphology_cosine = frozen_outputs["evidence_morphology_alignment_cosine"].clamp(-1.0, 1.0)
        valid_mask = frozen_outputs["evidence_valid_mechanism"].clamp(0.0, 1.0)
        mechanism_valid_norm = frozen_outputs["evidence_valid_mechanism_norm"].clamp_min(0.0) * valid_mask
        return {
            "normalized_support_strength": support * valid_mask,
            "normalized_opposition_strength": opposition * valid_mask,
            "normalized_uncertainty_strength": uncertainty,
            "normalized_conflict_score": conflict,
            "normalized_temporal_conflict_score": temporal_conflict,
            "morphology_alignment_cosine": morphology_cosine,
            "valid_mechanism_evidence_norm": mechanism_valid_norm,
            "valid_mechanism_evidence_mask": valid_mask,
        }

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            frozen_outputs = self.frozen_c17(batch)

        evidence = self._normalize_evidence(frozen_outputs)
        q_support = self.support_calibrator(evidence["normalized_support_strength"])
        q_opposition = self.opposition_calibrator(evidence["normalized_opposition_strength"])
        evidence_gap = q_support - q_opposition
        evidence_polarity = torch.tanh(evidence_gap / max(self.polarity_temperature, 1e-6))
        evidence_confidence = (
            evidence_polarity.abs()
            * (1.0 - evidence["normalized_uncertainty_strength"])
            * (1.0 - evidence["normalized_conflict_score"])
        ).clamp(0.0, 1.0)
        frozen_c17_logit = frozen_outputs["logit"]
        magnitude_features = torch.stack(
            [
                evidence_polarity.abs(),
                evidence_confidence,
                frozen_c17_logit.abs(),
                evidence["valid_mechanism_evidence_norm"],
            ],
            dim=-1,
        )
        correction_magnitude = self.magnitude_head(magnitude_features)
        # Keep the scale non-negative without creating a dead branch if AdamW
        # crosses zero from the tiny equivalence-preserving initialization.
        magnitude_scale = self.residual_scale.abs().clamp(max=1.0)
        effective_correction_magnitude = correction_magnitude * magnitude_scale
        delta_c19 = evidence_polarity * evidence_confidence * effective_correction_magnitude
        final_logit = frozen_c17_logit + delta_c19
        return {
            **frozen_outputs,
            "frozen_c17_logit": frozen_c17_logit,
            "frozen_c17_prob": torch.sigmoid(frozen_c17_logit),
            "q_support": q_support,
            "q_opposition": q_opposition,
            "evidence_gap": evidence_gap,
            "evidence_polarity": evidence_polarity,
            "evidence_confidence": evidence_confidence,
            "correction_magnitude": correction_magnitude,
            "magnitude_scale": magnitude_scale.expand_as(correction_magnitude),
            "effective_correction_magnitude": effective_correction_magnitude,
            "normalized_support_strength": evidence["normalized_support_strength"],
            "normalized_opposition_strength": evidence["normalized_opposition_strength"],
            "normalized_uncertainty_strength": evidence["normalized_uncertainty_strength"],
            "normalized_conflict_score": evidence["normalized_conflict_score"],
            "normalized_temporal_conflict_score": evidence["normalized_temporal_conflict_score"],
            "morphology_alignment_cosine": evidence["morphology_alignment_cosine"],
            "valid_mechanism_evidence_norm": evidence["valid_mechanism_evidence_norm"],
            "valid_mechanism_evidence_mask": evidence["valid_mechanism_evidence_mask"],
            "logit": final_logit,
            "prob": torch.sigmoid(final_logit),
            "delta_c19": delta_c19,
        }
