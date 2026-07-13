from __future__ import annotations

import math
import re
from typing import Dict, Iterable, List, Sequence

import torch
import torch.nn.functional as F
from torch import nn


SUPPORT_TERMS = (
    "弥漫性改变",
    "弥漫性病变",
    "桥本样改变",
    "实质回声不均",
    "实质回声粗糙",
    "腺体回声不均",
    "弥漫性回声改变",
    "回声不均",
    "回声欠均",
    "回声欠均匀",
    "回声减低",
    "低回声",
)

DIAGNOSTIC_HINT_TERMS = ("桥本甲状腺炎", "慢性淋巴细胞性甲状腺炎", "桥本", "HT")

OPPOSITION_TERMS = (
    "甲状腺未见明显异常",
    "双侧甲状腺未见明显异常",
    "甲状腺实质回声均匀",
    "实质回声均匀",
    "回声均匀",
    "未见明显弥漫性改变",
    "未见弥漫性改变",
    "无明显弥漫性改变",
    "未见明显弥漫性病变",
    "未见明显甲状腺弥漫性病变",
    "未见明显桥本表现",
    "未见桥本样改变",
    "未提示桥本样改变",
)

UNCERTAINTY_TERMS = ("考虑", "可疑", "倾向", "建议结合", "建议复查", "不能除外", "待排")

NONSPECIFIC_TERMS = (
    "结节",
    "低回声结节",
    "无回声",
    "边界清",
    "边界清晰",
    "形态规则",
    "椭圆",
    "囊",
    "未见明显血流",
    "未见血流",
    "后方回声无明显改变",
    "内部回声均匀",
)

NEGATION_TRIGGERS = ("未见明显", "无明显", "未见明确", "未发现", "未提示", "未显示", "未见", "没有", "否认", "无")

TEXT_MASK_KEYS = (
    "text_support_mask",
    "text_opposition_mask",
    "text_uncertainty_mask",
    "text_nonspecific_mask",
    "text_diagnostic_hint_mask",
    "text_latest_mask",
    "text_history_mask",
    "text_full_report_mask",
)


def _term_occurrences(text: str, term: str) -> Iterable[tuple[int, int]]:
    if term.upper() == "HT":
        for match in re.finditer(r"(?<![A-Z])HT(?![A-Z])", text.upper()):
            yield match.start(), match.end()
        return
    start = 0
    while True:
        index = text.find(term, start)
        if index < 0:
            return
        yield index, index + len(term)
        start = index + max(len(term), 1)


def _is_negated(text: str, start: int, window: int = 10) -> bool:
    prefix = text[max(0, start - window) : start]
    return any(trigger in prefix for trigger in NEGATION_TRIGGERS)


def _mark(mask: torch.Tensor, start: int, end: int, visible_chars: int) -> None:
    left = max(0, min(start, visible_chars)) + 1
    right = max(left, min(end, visible_chars) + 1)
    mask[left:right] = 1.0


def _mark_terms(
    text: str,
    terms: Sequence[str],
    mask: torch.Tensor,
    visible_chars: int,
    reject_negated: bool = False,
    negated_target: torch.Tensor | None = None,
) -> None:
    for term in terms:
        for start, end in _term_occurrences(text, term):
            if reject_negated and _is_negated(text, start):
                if negated_target is not None:
                    _mark(negated_target, max(0, start - 10), end, visible_chars)
                continue
            _mark(mask, start, end, visible_chars)


def _section_span(text: str, marker: str, end_markers: Sequence[str]) -> tuple[int, int] | None:
    start = text.find(marker)
    if start < 0:
        return None
    marker_end = text.find("]", start)
    if marker_end < 0:
        return None
    content_start = marker_end + 1
    candidates = [text.find(end_marker, content_start) for end_marker in end_markers]
    candidates = [candidate for candidate in candidates if candidate >= 0]
    return content_start, min(candidates) if candidates else len(text)


def build_text_evidence_masks(text: str, max_length: int) -> Dict[str, torch.Tensor]:
    """Build character-position pooling masks without creating new labels."""
    normalized = str(text or "").strip()
    visible_chars = max(max_length - 2, 1)
    visible_text = normalized[:visible_chars]
    masks = {key: torch.zeros(max_length, dtype=torch.float32) for key in TEXT_MASK_KEYS}

    _mark_terms(
        visible_text,
        SUPPORT_TERMS,
        masks["text_support_mask"],
        visible_chars,
        reject_negated=True,
        negated_target=masks["text_opposition_mask"],
    )
    _mark_terms(visible_text, OPPOSITION_TERMS, masks["text_opposition_mask"], visible_chars)
    _mark_terms(visible_text, UNCERTAINTY_TERMS, masks["text_uncertainty_mask"], visible_chars)
    _mark_terms(visible_text, NONSPECIFIC_TERMS, masks["text_nonspecific_mask"], visible_chars)
    _mark_terms(
        visible_text,
        DIAGNOSTIC_HINT_TERMS,
        masks["text_diagnostic_hint_mask"],
        visible_chars,
        reject_negated=True,
        negated_target=masks["text_opposition_mask"],
    )

    latest = _section_span(visible_text, "[C13_LATEST_THYROID", ("[C13_HISTORY_THYROID]", "[C13_FULL_REPORT]"))
    history = _section_span(visible_text, "[C13_HISTORY_THYROID]", ("[C13_FULL_REPORT]",))
    full_report = _section_span(visible_text, "[C13_FULL_REPORT]", ())
    for span, key in (
        (latest, "text_latest_mask"),
        (history, "text_history_mask"),
        (full_report, "text_full_report_mask"),
    ):
        if span is not None:
            _mark(masks[key], span[0], span[1], visible_chars)
    if full_report is None and visible_text:
        _mark(masks["text_full_report_mask"], 0, len(visible_text), visible_chars)
    return masks


def _masked_softmax(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask.bool()
    safe = valid.clone()
    empty = ~safe.any(dim=-1)
    if bool(empty.any().item()):
        safe[empty, 0] = True
    masked = scores.masked_fill(~safe, torch.finfo(scores.dtype).min)
    weights = torch.softmax(masked, dim=-1)
    return weights * safe.to(weights.dtype)


def _masked_mean(nodes: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    weights = valid.to(nodes.dtype).unsqueeze(-1)
    return (nodes * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


class ImageMorphologyEvidenceProjector(nn.Module):
    slot_names = ("diffuse", "texture", "structural", "nonspecific", "global")

    def __init__(self, hidden_dim: int, dropout: float, num_heads: int = 4) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, len(self.slot_names), hidden_dim) * 0.02)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, image_tokens: torch.Tensor, image_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch = image_tokens.shape[0]
        valid_tokens = image_mask.bool()
        safe_tokens = valid_tokens.clone()
        no_image = ~safe_tokens.any(dim=1)
        if bool(no_image.any().item()):
            safe_tokens[no_image, 0] = True
        queries = self.queries.expand(batch, -1, -1)
        attended, weights = self.attn(
            queries,
            image_tokens,
            image_tokens,
            key_padding_mask=~safe_tokens,
            need_weights=True,
            average_attn_weights=False,
        )
        nodes = self.out(attended)
        available = (~no_image).unsqueeze(1).expand(-1, len(self.slot_names))
        nodes = nodes * available.unsqueeze(-1).to(nodes.dtype)
        mean_weights = weights.mean(dim=1)
        entropy = -(mean_weights.clamp_min(1e-8) * mean_weights.clamp_min(1e-8).log()).sum(dim=-1).mean(dim=-1)
        return {
            "nodes": nodes,
            "valid": available,
            "attention_entropy": entropy,
            "slot_norm_mean": nodes.norm(dim=-1).mean(dim=-1),
        }


class TextEvidenceRoleProjector(nn.Module):
    role_names = ("support", "opposition", "uncertainty", "nonspecific", "temporal", "global")

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.randn(len(self.role_names), hidden_dim) * 0.02)
        self.out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.temporal_proj = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )
        self.temporal_conflict = nn.Linear(hidden_dim, 1)

    def _pool(
        self,
        tokens: torch.Tensor,
        attention_mask: torch.Tensor,
        guided_mask: torch.Tensor,
        query: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scores = torch.einsum("blh,h->bl", tokens, query) / math.sqrt(tokens.shape[-1])
        base_weights = _masked_softmax(scores, attention_mask)
        guided = attention_mask.bool() & guided_mask.bool()
        guided_present = guided.any(dim=-1)
        guided_weights = _masked_softmax(scores, guided)
        weights = torch.where(guided_present.unsqueeze(-1), guided_weights, base_weights)
        node = torch.einsum("bl,blh->bh", weights, tokens)
        guided_mass = (base_weights * guided.to(base_weights.dtype)).sum(dim=-1)
        return node, guided_mass, guided_present

    def forward(self, text_tokens: torch.Tensor, attention_mask: torch.Tensor, masks: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        support_mask = torch.maximum(masks["text_support_mask"], masks["text_diagnostic_hint_mask"])
        role_masks = [
            support_mask,
            masks["text_opposition_mask"],
            masks["text_uncertainty_mask"],
            masks["text_nonspecific_mask"],
            torch.maximum(masks["text_latest_mask"], masks["text_history_mask"]),
            masks["text_full_report_mask"],
        ]
        pooled: List[torch.Tensor] = []
        masses: List[torch.Tensor] = []
        present: List[torch.Tensor] = []
        for index, role_mask in enumerate(role_masks):
            node, mass, has_guidance = self._pool(text_tokens, attention_mask, role_mask, self.queries[index])
            pooled.append(node)
            masses.append(mass)
            present.append(has_guidance if index < len(role_masks) - 1 else attention_mask.bool().any(dim=-1))

        latest_node, _, latest_present = self._pool(
            text_tokens, attention_mask, masks["text_latest_mask"], self.queries[4]
        )
        history_node, _, history_present = self._pool(
            text_tokens, attention_mask, masks["text_history_mask"], self.queries[4]
        )
        temporal = self.temporal_proj(
            torch.cat([latest_node, history_node, torch.abs(latest_node - history_node), latest_node * history_node], dim=-1)
        )
        temporal_present = latest_present | history_present
        pooled[4] = torch.where(temporal_present.unsqueeze(-1), temporal, pooled[4])
        present[4] = temporal_present
        nodes = self.out(torch.stack(pooled, dim=1))
        latest_role_node = self.out(latest_node)
        history_role_node = self.out(history_node)
        guidance_present = torch.stack(present, dim=1)
        text_available = attention_mask.bool().any(dim=-1, keepdim=True)
        valid = text_available.expand(-1, len(self.role_names))
        morphology_guided = attention_mask.bool() & torch.maximum(
            masks["text_support_mask"], masks["text_nonspecific_mask"]
        ).bool()
        masses_tensor = torch.stack(masses, dim=1)
        return {
            "nodes": nodes,
            "valid": valid,
            "guidance_present": guidance_present,
            "morphology_guidance_present": morphology_guided.any(dim=-1),
            "guided_attention_mass": masses_tensor,
            "temporal_conflict_score": torch.sigmoid(self.temporal_conflict(temporal).squeeze(-1))
            * temporal_present.to(temporal.dtype),
            "temporal_available": temporal_present,
            "latest_role_node": latest_role_node,
            "history_role_node": history_role_node,
            "latest_available": latest_present,
            "history_available": history_present,
            "role_norm_mean": nodes.norm(dim=-1).mean(dim=-1),
        }


class BioEvidenceProjector(nn.Module):
    group_names = ("other_observed", "immune_observed", "function_observed")
    group_indices = ((0, 1), (2, 5), (3, 4, 6))

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.group_proj = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.LayerNorm(hidden_dim),
                )
                for _ in self.group_names
            ]
        )

    def forward(self, bio_tokens: torch.Tensor, bio_missing_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        tokens = self.input_norm(bio_tokens)
        observed = ~bio_missing_mask.bool()
        nodes: List[torch.Tensor] = []
        valid: List[torch.Tensor] = []
        for projector, indices in zip(self.group_proj, self.group_indices):
            index = torch.tensor(indices, device=tokens.device)
            group_tokens = tokens.index_select(1, index)
            group_valid = observed.index_select(1, index)
            group_node = _masked_mean(group_tokens, group_valid)
            has_group = group_valid.any(dim=-1)
            group_node = projector(group_node) * has_group.unsqueeze(-1).to(group_node.dtype)
            nodes.append(group_node)
            valid.append(has_group)
        stacked = torch.stack(nodes, dim=1)
        return {
            "nodes": stacked,
            "valid": torch.stack(valid, dim=1),
            "norm_mean": stacked.norm(dim=-1).mean(dim=-1),
            "valid_fraction": observed.float().mean(dim=-1),
        }


class EvidenceRoleScorer(nn.Module):
    role_names = ("support", "opposition", "uncertainty")

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, len(self.role_names)),
        )

    def forward(self, nodes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.net(nodes)
        return logits, torch.softmax(logits, dim=-1)


class HTMechanismRelationLayer(nn.Module):
    mechanism_names = ("M1_morphology", "M2_immune", "M3_function", "M4_opposition", "M5_temporal")

    def __init__(self, hidden_dim: int, dropout: float, num_heads: int = 4) -> None:
        super().__init__()
        relation_names = (
            "image_morphology",
            "text_morphology",
            "bio_immune",
            "bio_function",
            "text_opposition",
            "text_temporal",
            "text_global",
            "bio_other",
        )
        self.relations = nn.ModuleDict(
            {
                name: nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout))
                for name in relation_names
            }
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in self.mechanism_names])
        self.disease_query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.disease_attn = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.disease_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        image_nodes: torch.Tensor,
        image_valid: torch.Tensor,
        text_nodes: torch.Tensor,
        text_valid: torch.Tensor,
        text_morphology_guidance_present: torch.Tensor,
        bio_nodes: torch.Tensor,
        bio_valid: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        image_available = image_valid.any(dim=-1)
        image_morph = self.relations["image_morphology"](_masked_mean(image_nodes, image_valid))
        image_morph = image_morph * image_available.unsqueeze(-1).to(image_morph.dtype)
        text_morph_valid = torch.stack([text_valid[:, 0], text_valid[:, 3]], dim=1)
        text_morph = self.relations["text_morphology"](
            _masked_mean(torch.stack([text_nodes[:, 0], text_nodes[:, 3]], dim=1), text_morph_valid)
        )
        text_morph_available = text_morph_valid.any(dim=-1)
        text_morph = text_morph * text_morph_available.unsqueeze(-1).to(text_morph.dtype)
        m1_valid = image_available | text_morph_available
        m1 = self.norms[0](image_morph + text_morph) * m1_valid.unsqueeze(-1).to(image_morph.dtype)

        m2_valid = bio_valid[:, 1]
        m2 = self.norms[1](self.relations["bio_immune"](bio_nodes[:, 1])) * m2_valid.unsqueeze(-1).to(image_morph.dtype)
        m3_valid = bio_valid[:, 2]
        m3 = self.norms[2](self.relations["bio_function"](bio_nodes[:, 2])) * m3_valid.unsqueeze(-1).to(image_morph.dtype)
        m4_valid = text_valid[:, 1]
        m4 = self.norms[3](self.relations["text_opposition"](text_nodes[:, 1])) * m4_valid.unsqueeze(-1).to(image_morph.dtype)
        m5_valid = text_valid[:, 4]
        m5 = self.norms[4](self.relations["text_temporal"](text_nodes[:, 4])) * m5_valid.unsqueeze(-1).to(image_morph.dtype)

        states = torch.stack([m1, m2, m3, m4, m5], dim=1)
        valid = torch.stack([m1_valid, m2_valid, m3_valid, m4_valid, m5_valid], dim=1)
        safe_valid = valid.clone()
        empty = ~safe_valid.any(dim=-1)
        if bool(empty.any().item()):
            safe_valid[empty, 0] = True
        query = self.disease_query.expand(states.shape[0], -1, -1)
        disease, attention = self.disease_attn(
            query,
            states,
            states,
            key_padding_mask=~safe_valid,
            need_weights=True,
        )
        text_global_valid = text_valid[:, 5]
        context = self.relations["text_global"](text_nodes[:, 5]) * text_global_valid.unsqueeze(-1).to(states.dtype)
        context = context + self.relations["bio_other"](bio_nodes[:, 0]) * bio_valid[:, 0].unsqueeze(-1).to(context.dtype)
        disease = self.disease_norm(disease.squeeze(1) + context)
        morph_valid = image_available & text_morphology_guidance_present
        morph_cosine = F.cosine_similarity(image_morph, text_morph, dim=-1)
        return {
            "states": states,
            "valid": valid,
            "disease_state": disease,
            "attention": attention.squeeze(1),
            "morphology_image": image_morph,
            "morphology_text": text_morph,
            "morphology_valid": morph_valid,
            "morphology_cosine": morph_cosine,
        }


class EvidenceConflictAggregator(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.reliability = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))
        self.conflict = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )
        self.conflict_score = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        nodes: torch.Tensor,
        valid: torch.Tensor,
        role_probs: torch.Tensor,
        modality_slices: Sequence[slice],
    ) -> Dict[str, torch.Tensor]:
        reliability = torch.sigmoid(self.reliability(nodes).squeeze(-1)) * valid.to(nodes.dtype)
        role_weights = reliability.unsqueeze(-1) * role_probs
        evidence_states: List[torch.Tensor] = []
        strengths: List[torch.Tensor] = []
        for role_index in range(3):
            weights = role_weights[:, :, role_index]
            evidence_states.append(torch.einsum("bn,bnh->bh", weights, nodes) / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6))
            strengths.append(weights.sum(dim=-1) / valid.to(nodes.dtype).sum(dim=-1).clamp_min(1.0))
        support, opposition, uncertainty = evidence_states
        conflict = self.conflict(torch.cat([support, opposition, torch.abs(support - opposition), support * opposition], dim=-1))
        any_evidence = valid.any(dim=-1)
        conflict = conflict * any_evidence.unsqueeze(-1).to(conflict.dtype)
        conflict_score = torch.sigmoid(self.conflict_score(conflict).squeeze(-1))
        conflict_score = conflict_score * any_evidence.to(conflict_score.dtype)

        modality_raw: List[torch.Tensor] = []
        for modality_slice in modality_slices:
            modality_valid = valid[:, modality_slice]
            modality_reliability = reliability[:, modality_slice]
            modality_raw.append(
                modality_reliability.sum(dim=-1) / modality_valid.to(nodes.dtype).sum(dim=-1).clamp_min(1.0)
            )
        modality_raw_tensor = torch.stack(modality_raw, dim=-1)
        modality_available = torch.stack([valid[:, item].any(dim=-1) for item in modality_slices], dim=-1)
        modality_weights = modality_raw_tensor * modality_available.to(nodes.dtype)
        modality_weights = modality_weights / modality_weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        role_entropy = -(role_probs.clamp_min(1e-8) * role_probs.clamp_min(1e-8).log()).sum(dim=-1)
        role_entropy = (role_entropy * valid.to(nodes.dtype)).sum(dim=-1) / valid.to(nodes.dtype).sum(dim=-1).clamp_min(1.0)
        return {
            "support": support,
            "opposition": opposition,
            "uncertainty": uncertainty,
            "conflict": conflict,
            "conflict_score": conflict_score,
            "strengths": torch.stack(strengths, dim=-1),
            "reliability": reliability,
            "modality_weights": modality_weights,
            "role_entropy": role_entropy,
        }


class MechanismEvidenceAggregationHead(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.support_score = nn.Linear(hidden_dim, 1)
        self.opposition_score = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 5 + 1, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, mechanism_state: torch.Tensor, aggregate: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        q_support = self.support_score(aggregate["support"]).squeeze(-1)
        q_opposition = self.opposition_score(aggregate["opposition"]).squeeze(-1)
        representation = torch.cat(
            [
                mechanism_state,
                aggregate["support"],
                aggregate["opposition"],
                aggregate["uncertainty"],
                aggregate["conflict"],
                aggregate["conflict_score"].unsqueeze(-1),
            ],
            dim=-1,
        )
        logit = self.classifier(representation).squeeze(-1)
        return {
            "logit": logit,
            "prob": torch.sigmoid(logit),
            "q_support": q_support,
            "q_opposition": q_opposition,
            "state_margin": q_support - q_opposition,
        }


DiseaseStateAlignmentHead = MechanismEvidenceAggregationHead


class MechanismEvidenceAlignment(nn.Module):
    """Disease-mechanism and evidence-aware alignment over C13 encoder tokens."""

    def __init__(self, hidden_dim: int, dropout: float, num_heads: int = 4) -> None:
        super().__init__()
        self.image = ImageMorphologyEvidenceProjector(hidden_dim, dropout, num_heads=num_heads)
        self.text = TextEvidenceRoleProjector(hidden_dim, dropout)
        self.bio = BioEvidenceProjector(hidden_dim, dropout)
        self.role_scorer = EvidenceRoleScorer(hidden_dim, dropout)
        self.mechanisms = HTMechanismRelationLayer(hidden_dim, dropout, num_heads=num_heads)
        self.aggregator = EvidenceConflictAggregator(hidden_dim, dropout)
        self.head = MechanismEvidenceAggregationHead(hidden_dim, dropout)

    @staticmethod
    def _modality_role_mean(role_probs: torch.Tensor, valid: torch.Tensor, item: slice) -> torch.Tensor:
        return _masked_mean(role_probs[:, item], valid[:, item])

    def forward(
        self,
        image_tokens: torch.Tensor,
        image_mask: torch.Tensor,
        text_tokens: torch.Tensor,
        text_attention_mask: torch.Tensor,
        bio_tokens: torch.Tensor,
        bio_missing_mask: torch.Tensor,
        text_masks: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        image = self.image(image_tokens, image_mask)
        text = self.text(text_tokens, text_attention_mask, text_masks)
        bio = self.bio(bio_tokens, bio_missing_mask)
        nodes = torch.cat([image["nodes"], text["nodes"], bio["nodes"]], dim=1)
        valid = torch.cat([image["valid"], text["valid"], bio["valid"]], dim=1)
        role_logits, role_probs = self.role_scorer(nodes)
        _, latest_role_probs = self.role_scorer(text["latest_role_node"].unsqueeze(1))
        _, history_role_probs = self.role_scorer(text["history_role_node"].unsqueeze(1))
        mechanisms = self.mechanisms(
            image["nodes"],
            image["valid"],
            text["nodes"],
            text["valid"],
            text["morphology_guidance_present"],
            bio["nodes"],
            bio["valid"],
        )
        slices = (slice(0, 5), slice(5, 11), slice(11, 14))
        aggregate = self.aggregator(nodes, valid, role_probs, slices)
        head = self.head(mechanisms["disease_state"], aggregate)

        image_reliability = _masked_mean(
            aggregate["reliability"][:, slices[0]].unsqueeze(-1), image["valid"]
        ).squeeze(-1)
        text_morph_indices = torch.tensor([0, 3], device=nodes.device)
        text_morph_reliability = _masked_mean(
            aggregate["reliability"][:, slices[1]].index_select(1, text_morph_indices).unsqueeze(-1),
            text["valid"].index_select(1, text_morph_indices),
        ).squeeze(-1)
        pair_reliability = 0.5 * (image_reliability + text_morph_reliability)
        morph_weight = (
            mechanisms["morphology_valid"].to(nodes.dtype)
            * pair_reliability
            * (1.0 - aggregate["conflict_score"].detach())
        )
        if bool((morph_weight > 0).any().item()):
            mechanism_alignment_loss = (
                (1.0 - mechanisms["morphology_cosine"]) * morph_weight
            ).sum() / morph_weight.sum().clamp_min(1e-6)
        else:
            mechanism_alignment_loss = nodes.sum() * 0.0
        support_opposition_cosine = F.cosine_similarity(aggregate["support"], aggregate["opposition"], dim=-1)
        role_separation_loss = support_opposition_cosine.square().mean()
        image_roles = self._modality_role_mean(role_probs, valid, slices[0])
        text_roles = self._modality_role_mean(role_probs, valid, slices[1])
        bio_roles = self._modality_role_mean(role_probs, valid, slices[2])

        return {
            **head,
            "mea_mechanism_state": mechanisms["disease_state"],
            "mea_mechanism_nodes": mechanisms["states"],
            "mea_mechanism_valid": mechanisms["valid"],
            "mea_support_state": aggregate["support"],
            "mea_opposition_state": aggregate["opposition"],
            "mea_uncertainty_state": aggregate["uncertainty"],
            "mea_conflict_state": aggregate["conflict"],
            "mea_strengths": aggregate["strengths"],
            "conflict_suppression": (1.0 - aggregate["conflict_score"]).clamp(0.0, 1.0),
            "mea_mechanism_alignment_loss": mechanism_alignment_loss,
            "mea_role_separation_loss": role_separation_loss,
            "patient_support_strength": head["q_support"],
            "patient_opposition_strength": head["q_opposition"],
            "patient_uncertainty_strength": aggregate["strengths"][:, 2],
            "patient_conflict_score": aggregate["conflict_score"],
            "evidence_support_strength": head["q_support"],
            "evidence_opposition_strength": head["q_opposition"],
            "evidence_uncertainty_strength": aggregate["strengths"][:, 2],
            "evidence_conflict_score": aggregate["conflict_score"],
            "evidence_temporal_conflict_score": text["temporal_conflict_score"],
            "evidence_morphology_alignment_cosine": mechanisms["morphology_cosine"],
            "evidence_valid_mechanism_norm": mechanisms["disease_state"].norm(dim=-1),
            "evidence_valid_mechanism": mechanisms["valid"][:, 1:5].any(dim=-1).to(nodes.dtype),
            "image_support_score": image_roles[:, 0],
            "image_opposition_score": image_roles[:, 1],
            "image_uncertainty_score": image_roles[:, 2],
            "text_support_score": text_roles[:, 0],
            "text_opposition_score": text_roles[:, 1],
            "text_uncertainty_score": text_roles[:, 2],
            "bio_support_score": bio_roles[:, 0],
            "bio_opposition_score": bio_roles[:, 1],
            "bio_uncertainty_score": bio_roles[:, 2],
            "image_evidence_weight": aggregate["modality_weights"][:, 0],
            "text_evidence_weight": aggregate["modality_weights"][:, 1],
            "bio_evidence_weight": aggregate["modality_weights"][:, 2],
            "text_support_attention_mass": text["guided_attention_mass"][:, 0],
            "text_opposition_attention_mass": text["guided_attention_mass"][:, 1],
            "text_uncertainty_attention_mass": text["guided_attention_mass"][:, 2],
            "text_temporal_conflict_score": text["temporal_conflict_score"],
            "text_temporal_available": text["temporal_available"].to(nodes.dtype),
            "text_latest_support_score": latest_role_probs[:, 0, 0],
            "text_latest_opposition_score": latest_role_probs[:, 0, 1],
            "text_latest_available": text["latest_available"].to(nodes.dtype),
            "text_history_support_score": history_role_probs[:, 0, 0],
            "text_history_opposition_score": history_role_probs[:, 0, 1],
            "text_history_available": text["history_available"].to(nodes.dtype),
            "image_evidence_attention_entropy": image["attention_entropy"],
            "image_evidence_slot_norm_mean": image["slot_norm_mean"],
            "text_role_norm_mean": text["role_norm_mean"],
            "bio_evidence_norm_mean": bio["norm_mean"],
            "bio_valid_fraction": bio["valid_fraction"],
            "bio_evidence_reliability": _masked_mean(
                aggregate["reliability"][:, slices[2]].unsqueeze(-1), bio["valid"]
            ).squeeze(-1),
            "morphology_alignment_cosine": mechanisms["morphology_cosine"],
            "morphology_alignment_available": mechanisms["morphology_valid"].to(nodes.dtype),
            "support_opposition_cosine": support_opposition_cosine,
            "mechanism_state_norm": mechanisms["disease_state"].norm(dim=-1),
            "mechanism_attention_max": mechanisms["attention"].max(dim=-1).values,
            "evidence_role_entropy": aggregate["role_entropy"],
            "evidence_role_prob_sum_error": (role_probs.sum(dim=-1) - 1.0).abs().max(dim=-1).values,
            "evidence_reliability_mean": (
                aggregate["reliability"].sum(dim=-1) / valid.to(nodes.dtype).sum(dim=-1).clamp_min(1.0)
            ),
            "mea_role_logits": role_logits,
            "mea_role_probs": role_probs,
            "mea_node_valid": valid,
        }
