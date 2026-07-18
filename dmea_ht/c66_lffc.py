"""C66 fold-local public-backbone source learning and CBPI adaptation models."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from dmea_ht.c61_cbpi import C61CBPIModel
from dmea_ht.c41_melr import _masked_mean
from dmea_ht.mechanism_evidence_alignment import (
    BioEvidenceProjector,
    ImageMorphologyEvidenceProjector,
    TEXT_MASK_KEYS,
    TextEvidenceRoleProjector,
    build_text_evidence_masks,
)
from dmea_ht.models import BioEncoder
from dmea_ht.visit_data import VisitPatientDataset, collate_visit_batch


def _load_state(path: Path) -> Mapping[str, torch.Tensor]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"C66 public image weights are not a state mapping: {path}")
    return payload


class C66ImageEncoder(nn.Module):
    """ImageNet-initialized ResNet-50 followed by a fold-local projection."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        from torchvision.models import resnet50

        model_cfg = dict(config["model"])
        generic = dict(config["generic_initialization"])["image"]
        weight_path = Path(str(generic["local_weight_path"]))
        if not weight_path.exists():
            raise FileNotFoundError(f"C66 public ResNet50 weight is missing: {weight_path}")
        backbone = resnet50(weights=None)
        result = backbone.load_state_dict(_load_state(weight_path), strict=True)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError("C66 ResNet50 public initialization is not exact")
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        hidden_dim = int(model_cfg["hidden_dim"])
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(float(model_cfg["dropout"])),
            nn.Linear(2048, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, images: torch.Tensor, image_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, image_count, channels, height, width = images.shape
        flat = images.reshape(batch * image_count, channels, height, width)
        tokens = self.proj(self.backbone(flat)).reshape(batch, image_count, -1)
        weights = image_mask.unsqueeze(-1).to(tokens.dtype)
        global_token = (tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return tokens, global_token


class C66TextEncoder(nn.Module):
    """Public Stable Diffusion v1.5 CLIP text encoder with a fold-local projection."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        from transformers import CLIPTextModel

        model_cfg = dict(config["model"])
        generic = dict(config["generic_initialization"])["text"]
        text_encoder_path = Path(str(generic["text_encoder_path"]))
        if not text_encoder_path.exists():
            raise FileNotFoundError(f"C66 public CLIP text encoder is missing: {text_encoder_path}")
        self.backbone = CLIPTextModel.from_pretrained(
            str(text_encoder_path), local_files_only=True, use_safetensors=True
        )
        hidden_dim = int(model_cfg["hidden_dim"])
        source_hidden = int(self.backbone.config.hidden_size)
        self.proj = nn.Sequential(
            nn.Dropout(float(model_cfg["dropout"])),
            nn.Linear(source_hidden, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.backbone(input_ids=input_ids.long(), attention_mask=attention_mask.long())
        tokens = self.proj(outputs.last_hidden_state)
        weights = attention_mask.unsqueeze(-1).to(tokens.dtype)
        global_token = (tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return tokens, global_token


class C66PublicModalitySources(nn.Module):
    """Public image/text modules plus seed-random bio/projector modules."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        model_cfg = dict(config["model"])
        hidden_dim = int(model_cfg["hidden_dim"])
        dropout = float(model_cfg["dropout"])
        self.initialization_type = "public_generic_image_text_with_seed_random_task_specific_modules"
        self.image_encoder = C66ImageEncoder(config)
        self.text_encoder = C66TextEncoder(config)
        self.bio_encoder = BioEncoder(int(model_cfg["bio_dim"]), hidden_dim, dropout)
        self.image_projector = ImageMorphologyEvidenceProjector(
            hidden_dim, dropout, num_heads=int(model_cfg["mea_num_heads"])
        )
        self.text_projector = TextEvidenceRoleProjector(hidden_dim, dropout)
        self.bio_projector = BioEvidenceProjector(hidden_dim, dropout)


def source_states(sources: C66PublicModalitySources, batch: Dict[str, torch.Tensor], hidden_dim: int) -> Dict[str, torch.Tensor]:
    batch_size, visits = batch["visit_mask"].shape
    images = batch["images"].flatten(0, 1)
    image_mask = batch["image_mask"].flatten(0, 1)
    clip_input_ids = batch["clip_input_ids"].flatten(0, 1)
    clip_attention_mask = batch["clip_attention_mask"].flatten(0, 1)
    bio_values = batch["bio_values"].flatten(0, 1)
    bio_missing = batch["bio_missing_mask"].flatten(0, 1)
    bio_abnormal = batch["bio_abnormal_flags"].flatten(0, 1)

    image_tokens, image_global = sources.image_encoder(images, image_mask)
    text_tokens, text_global = sources.text_encoder(clip_input_ids, clip_attention_mask)
    bio_tokens, bio_global, _, _ = sources.bio_encoder(bio_values, bio_missing, bio_abnormal)
    text_masks = {key: batch[key].flatten(0, 1) for key in TEXT_MASK_KEYS}
    image = sources.image_projector(image_tokens, image_mask)
    text = sources.text_projector(text_tokens, clip_attention_mask, text_masks)
    bio = sources.bio_projector(bio_tokens, bio_missing)

    image_nodes = image["nodes"].reshape(batch_size, visits, 5, hidden_dim)
    text_nodes = text["nodes"].reshape(batch_size, visits, 6, hidden_dim)
    bio_nodes = bio["nodes"].reshape(batch_size, visits, 3, hidden_dim)
    image_valid = image["valid"].reshape(batch_size, visits, 5).bool()
    text_valid = text["valid"].reshape(batch_size, visits, 6).bool()
    bio_valid = bio["valid"].reshape(batch_size, visits, 3).bool()
    image_state, image_available = _masked_mean(image_nodes, image_valid, dim=2)
    text_state, text_available = _masked_mean(text_nodes, text_valid, dim=2)
    bio_state, bio_available = _masked_mean(bio_nodes, bio_valid, dim=2)
    image_global = image_global.reshape(batch_size, visits, hidden_dim)
    text_global = text_global.reshape(batch_size, visits, hidden_dim)
    bio_global = bio_global.reshape(batch_size, visits, hidden_dim)
    visit_mask = batch["visit_mask"].bool()
    raw_valid = torch.stack(
        [image_mask.any(dim=1), clip_attention_mask.any(dim=1), (~bio_missing.bool()).any(dim=1)], dim=1
    ).reshape(batch_size, visits, 3)
    aligned_valid = torch.stack([image_available, text_available, bio_available], dim=2)
    states = torch.stack([image_global, text_global, bio_global, image_state, text_state, bio_state], dim=2)
    valid = torch.cat([raw_valid, aligned_valid], dim=2) & visit_mask.unsqueeze(-1)
    return {"states": states, "valid": valid, "visit_mask": visit_mask}


def source_visit_features(source: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    aligned = source["states"][:, :, 3:6]
    aligned_valid = source["valid"][:, :, 3:6].bool()
    weights = aligned_valid.to(aligned.dtype).unsqueeze(-1)
    mean_state = (aligned * weights).sum(dim=2) / weights.sum(dim=2).clamp_min(1.0)
    features = torch.cat([aligned[:, :, 0], aligned[:, :, 1], aligned[:, :, 2], mean_state], dim=-1)
    valid = aligned_valid.any(dim=2) & source["visit_mask"].bool()
    return features, valid


def build_source_evidence_stack(input_dim: int, instance_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(input_dim),
        nn.Linear(input_dim, instance_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.LayerNorm(instance_dim),
    )


def fixed_patient_statistics(
    tokens: torch.Tensor, valid: torch.Tensor, visit_mask: torch.Tensor
) -> Tuple[torch.Tensor, ...]:
    counts = visit_mask.sum(dim=1)
    positions = torch.arange(tokens.shape[1], device=tokens.device, dtype=counts.dtype).unsqueeze(0)
    latest_index = (counts - 1).clamp_min(0).unsqueeze(-1)
    latest_mask = visit_mask & positions.eq(latest_index)
    history_mask = visit_mask & ~latest_mask
    latest_weights = latest_mask.to(tokens.dtype)
    age = (latest_index - positions).clamp_min(0).to(tokens.dtype)
    history_kernel = (1.0 / torch.log2(age + 2.0)) * history_mask.to(tokens.dtype)
    history_weights = history_kernel / history_kernel.sum(dim=1, keepdim=True).clamp_min(1.0)

    valid_weights = valid.to(tokens.dtype)
    latest_effective = latest_weights * valid_weights
    latest_denominator = latest_effective.sum(dim=1, keepdim=True)
    latest = (tokens * latest_effective.unsqueeze(-1)).sum(dim=1) / latest_denominator.clamp_min(1.0)
    latest_valid = latest_denominator.squeeze(-1) > 0.0
    history_effective = history_weights * valid_weights
    history_denominator = history_effective.sum(dim=1, keepdim=True)
    history = (tokens * history_effective.unsqueeze(-1)).sum(dim=1) / history_denominator.clamp_min(1.0)
    history_valid = history_denominator.squeeze(-1) > 0.0
    delta = (latest - history) * (latest_valid & history_valid).unsqueeze(-1).to(tokens.dtype)
    centered = (tokens - history.unsqueeze(1)).pow(2)
    dispersion = (centered * history_effective.unsqueeze(-1)).sum(dim=1) / history_denominator.clamp_min(1.0)
    dispersion = dispersion.clamp_min(1e-8).sqrt() * history_valid.unsqueeze(-1).to(tokens.dtype)
    all_effective = valid_weights
    patient_denominator = all_effective.sum(dim=1, keepdim=True)
    patient_mean = (tokens * all_effective.unsqueeze(-1)).sum(dim=1) / patient_denominator.clamp_min(1.0)
    masked_tokens = tokens.masked_fill(~valid.unsqueeze(-1), torch.finfo(tokens.dtype).min)
    set_max = masked_tokens.max(dim=1).values
    patient_available = patient_denominator.squeeze(-1) > 0.0
    set_max = torch.where(patient_available.unsqueeze(-1), set_max, torch.zeros_like(set_max))
    return (
        latest,
        history,
        delta,
        dispersion,
        set_max,
        patient_mean,
        latest_weights,
        history_weights,
        latest_mask,
        history_mask,
        latest_valid,
        history_valid,
        patient_available,
    )


def source_bio_state(source: Mapping[str, torch.Tensor]) -> torch.Tensor:
    bio_state = source["states"][:, :, 5]
    bio_valid = source["valid"][:, :, 5].bool()
    weights = bio_valid.to(bio_state.dtype)
    return (bio_state * weights.unsqueeze(-1)).sum(dim=1) / weights.sum(dim=1, keepdim=True).clamp_min(1.0)


class C66SourceModel(nn.Module):
    """Fold-local source learner with a temporary source-only classifier."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        model_cfg = dict(config["model"])
        self.hidden_dim = int(model_cfg["hidden_dim"])
        self.instance_dim = int(model_cfg["instance_dim"])
        dropout = float(model_cfg["dropout"])
        self.sources = C66PublicModalitySources(config)
        self.source_evidence_stack = build_source_evidence_stack(
            self.hidden_dim * 4, self.instance_dim, dropout
        )
        self.source_patient_readout = nn.Sequential(
            nn.LayerNorm(self.instance_dim * 6),
            nn.Linear(self.instance_dim * 6, self.instance_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(self.instance_dim),
        )
        self.source_classifier = nn.Linear(self.instance_dim, 1)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        source = source_states(self.sources, batch, self.hidden_dim)
        features, valid = source_visit_features(source)
        tokens = self.source_evidence_stack(features) * valid.unsqueeze(-1).to(features.dtype)
        statistics = fixed_patient_statistics(tokens, valid, source["visit_mask"])
        latest, history, delta, dispersion, set_max, patient_mean = statistics[:6]
        latest_weights, history_weights, latest_mask, history_mask = statistics[6:10]
        latest_valid, history_valid, patient_available = statistics[10:]
        patient_state = self.source_patient_readout(
            torch.cat([latest, history, delta, dispersion, set_max, patient_mean], dim=-1)
        )
        logit = self.source_classifier(patient_state).squeeze(-1)
        evidence_tokens = torch.stack([latest, history, delta, dispersion], dim=1)
        evidence_valid = torch.stack([latest_valid, history_valid, patient_available, history_valid], dim=1)
        attention = evidence_valid.to(tokens.dtype)
        attention = attention / attention.sum(dim=1, keepdim=True).clamp_min(1.0)
        return {
            "logit": logit,
            "prob": torch.sigmoid(logit),
            "patient_state": patient_state,
            "attended_evidence": latest,
            "bio_state": source_bio_state(source),
            "evidence_tokens": evidence_tokens,
            "evidence_valid": evidence_valid,
            "latest_bio_valid": latest_valid,
            "history_bio_valid": history_valid,
            "latest_weights": latest_weights,
            "history_weights": history_weights,
            "latest_mask": latest_mask,
            "history_mask": history_mask,
            "attention": attention,
            "trajectory_available": valid,
            "instance_tokens": tokens,
            "instance_valid": valid,
        }


class C66CBPIModel(C61CBPIModel):
    """CBPI head coupled to the C66 fold-local public source stack."""

    def __init__(self, config: Dict[str, Any], seed: int) -> None:
        if not bool(config.get("from_base")) or config.get("initialization", {}).get("mode") != "from_base":
            raise RuntimeError("C66 CBPI construction requires clean from-base initialization")
        super().__init__(config, seed)
        model_cfg = dict(config["model"])
        self.sources = C66PublicModalitySources(config)
        self.source_evidence_stack = build_source_evidence_stack(
            int(model_cfg["hidden_dim"]) * 4, self.instance_dim, float(model_cfg["dropout"])
        )

    def _source_states(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return source_states(self.sources, batch, self.hidden_dim)

    def load_fold_local_source(self, payload: Mapping[str, Any]) -> None:
        source_state = payload.get("sources")
        stack_state = payload.get("source_evidence_stack")
        if not isinstance(source_state, Mapping) or not isinstance(stack_state, Mapping):
            raise RuntimeError("C66 source checkpoint does not contain fold-local source states")
        source_result = self.sources.load_state_dict(source_state, strict=True)
        stack_result = self.source_evidence_stack.load_state_dict(stack_state, strict=True)
        if source_result.missing_keys or source_result.unexpected_keys or stack_result.missing_keys or stack_result.unexpected_keys:
            raise RuntimeError("C66 fold-local source checkpoint scope mismatch")

    def configure_route(self, route: str) -> None:
        if route not in {"F", "E"}:
            raise RuntimeError(f"Unknown C66 route: {route}")
        source_trainable = route == "E"
        self.end_to_end = source_trainable
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(source_trainable or not name.startswith(("sources.", "source_evidence_stack.")))
        if not source_trainable:
            self.sources.eval()
            self.source_evidence_stack.eval()

    def train(self, mode: bool = True) -> "C66CBPIModel":
        super().train(mode)
        if not self.end_to_end:
            self.sources.eval()
            self.source_evidence_stack.eval()
        return self

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self.end_to_end:
            source = self._source_states(batch)
        else:
            with torch.no_grad():
                source = self._source_states(batch)
        source_features, source_valid = source_visit_features(source)
        if self.end_to_end:
            source_context = self.source_evidence_stack(source_features)
        else:
            with torch.no_grad():
                source_context = self.source_evidence_stack(source_features)
        source_context = source_context * source_valid.unsqueeze(-1).to(source_context.dtype)

        multimodal_features, visit_valid = self._multimodal_features(source)
        visit_valid = visit_valid & source_valid
        multimodal_token = self.multimodal_encoder(multimodal_features) + source_context
        bio_values = torch.nan_to_num(batch["bio_values"].float(), nan=0.0, posinf=0.0, neginf=0.0)
        bio_observed = ~batch["bio_missing_mask"].bool()
        bio_values = bio_values * bio_observed.to(bio_values.dtype)
        bio_nonlinear = torch.cat(
            [bio_values, torch.tanh(bio_values), bio_values * torch.tanh(bio_values)], dim=-1
        )
        continuous_bio_token = self.continuous_bio_encoder(bio_nonlinear)
        joint_features = torch.cat(
            [multimodal_token, continuous_bio_token, multimodal_token * continuous_bio_token], dim=-1
        )
        visit_tokens = self.joint_instance_encoder(joint_features)
        visit_tokens = visit_tokens * visit_valid.to(visit_tokens.dtype).unsqueeze(-1)
        statistics = self._fixed_patient_set_statistics(visit_tokens, visit_valid, source["visit_mask"])
        latest, history, delta, dispersion, set_max, patient_mean = statistics[:6]
        latest_weights, history_weights, latest_mask, history_mask = statistics[6:10]
        latest_valid, history_valid, patient_available = statistics[10:]
        patient_input = torch.cat([latest, history, delta, dispersion, set_max, patient_mean], dim=-1)
        patient_state = self.patient_readout(patient_input)
        logit = self.classifier(patient_state).squeeze(-1)
        evidence_tokens = torch.stack([latest, history, delta, dispersion], dim=1)
        evidence_valid = torch.stack([latest_valid, history_valid, patient_available, history_valid], dim=1)
        attention = evidence_valid.to(evidence_tokens.dtype)
        attention = attention / attention.sum(dim=1, keepdim=True).clamp_min(1.0)
        return {
            "logit": logit,
            "prob": torch.sigmoid(logit),
            "patient_state": patient_state,
            "attended_evidence": latest,
            "bio_state": source_bio_state(source),
            "evidence_tokens": evidence_tokens,
            "evidence_valid": evidence_valid,
            "latest_bio_valid": (latest_mask.unsqueeze(-1) & (~batch["bio_missing_mask"].bool())).any(dim=1),
            "history_bio_valid": (history_mask.unsqueeze(-1) & (~batch["bio_missing_mask"].bool())).any(dim=1),
            "latest_weights": latest_weights,
            "history_weights": history_weights,
            "latest_mask": latest_mask,
            "history_mask": history_mask,
            "attention": attention,
            "trajectory_available": visit_valid,
            "instance_tokens": visit_tokens,
            "instance_valid": visit_valid,
            "continuous_bio_token": continuous_bio_token,
            "fusion_state": patient_input,
        }


class C66VisitPatientDataset(VisitPatientDataset):
    """Development/Test visit dataset carrying local CLIP tokens for C66 only."""

    def __init__(self, *args: Any, clip_tokenizer_path: str | Path, clip_max_length: int, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        from transformers import CLIPTokenizer

        self.clip_tokenizer = CLIPTokenizer.from_pretrained(str(clip_tokenizer_path), local_files_only=True)
        self.clip_max_length = int(clip_max_length)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item = super().__getitem__(index)
        source_visits = list(self.rows[index].get("visits", []))
        if len(source_visits) != len(item["visits"]):
            raise RuntimeError("C66 visit dataset lost a visit during token construction")
        for visit_tensor, visit in zip(item["visits"], source_visits):
            text = str(visit.get("report_text", "") or "")
            encoded = self.clip_tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.clip_max_length,
                return_tensors="pt",
            )
            visit_tensor["clip_input_ids"] = encoded["input_ids"].squeeze(0).long()
            visit_tensor["clip_attention_mask"] = encoded["attention_mask"].squeeze(0).long()
            visit_tensor.update(build_text_evidence_masks(text, self.clip_max_length))
        return item


def collate_c66_visit_batch(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    output = collate_visit_batch(batch)
    max_visits = max(len(item["visits"]) for item in batch)
    for key in ("clip_input_ids", "clip_attention_mask"):
        example = batch[0]["visits"][0][key]
        padded = []
        for item in batch:
            value = torch.zeros((max_visits, *example.shape), dtype=example.dtype)
            for index, visit in enumerate(item["visits"]):
                value[index] = visit[key]
            padded.append(value)
        output[key] = torch.stack(padded)
    return output
