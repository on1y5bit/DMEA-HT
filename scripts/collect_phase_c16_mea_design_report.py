#!/usr/bin/env python3
"""Collect the C16-MEA input audit into a constrained design decision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


REQUIRED_FILES = [
    "c16_mea_current_model_path.md",
    "c16_mea_available_input_fields.csv",
    "c16_mea_available_bio_fields.csv",
    "c16_mea_bio_semantic_grouping.csv",
    "c16_mea_existing_text_evidence_terms.csv",
    "c16_mea_existing_temporal_fields.csv",
    "c16_mea_c14_diagnostic_mapping.csv",
    "c16_mea_shortcut_exclusion_map.csv",
    "c16_mea_design_feasibility.md",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", default="analysis_reports/phase_c16_mea_design")
    parser.add_argument("--require-pass", action="store_true")
    return parser.parse_args()


def fmt(value: Any, digits: int = 4) -> str:
    try:
        if value is None or pd.isna(value):
            return "NA"
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def frame_to_markdown(frame: pd.DataFrame, columns: List[str] | None = None) -> str:
    if frame.empty:
        return "_No rows._"
    view = frame[columns].copy() if columns else frame.copy()
    headers = [str(column) for column in view.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for _, row in view.iterrows():
        values = [fmt(row[column]).replace("|", "/").replace("\n", " ") for column in view.columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def check_map(summary: Dict[str, Any]) -> Dict[str, bool]:
    return {str(row["check"]): bool(row["pass"]) for row in summary.get("checks", [])}


def write_current_model_path(audit_dir: Path, summary: Dict[str, Any]) -> None:
    code = summary["code_path"]
    lines = [
        "# C16-MEA Current C13 Model Path",
        "",
        "## Frozen Input Contract",
        "",
        f"- Manifest: `{summary['manifest']}`",
        f"- Patients/rows: `{summary['unique_patients']}` / `{summary['manifest_rows']}`",
        f"- Model-visible report characters: `{summary['model_visible_report_chars']}`",
        f"- Split/label counts: `{json.dumps(summary['split_label_counts'], sort_keys=True)}`",
        "- Checkpoint selection remains validation AUC only; test remains reporting-only.",
        "",
        "## Current Path",
        "",
        "```text",
        *code["paths"],
        "```",
        "",
        "## Available Representations",
        "",
        f"- Per-image tokens and image global: `{code['image_tokens_available']}`",
        f"- Per-character text tokens and text global: `{code['text_tokens_available']}`",
        f"- Per-field bio tokens and bio global: `{code['bio_tokens_available']}`",
        f"- Existing patient anchor: `{code['patient_anchor_available']}`",
        f"- Existing classifier contributions: `{code['classifier_contributions_available']}`",
        f"- Character positions remain aligned with report positions: `{code['text_tokenization_is_character_position_preserving']}`",
        "",
        "## Existing Evidence Limitation",
        "",
        f"The current generic evidence roles are `{', '.join(code['current_generic_evidence_roles'])}`, but their role loss is zero: `{code['current_role_alignment_loss_is_zero']}`. They are not clinically grounded C16-MEA roles and must not be relabeled as such without a new evidence path.",
        "",
        "C16-MEA may consume encoder tokens and verified masks, but it must align evidence through documented HT mechanisms. It must not align raw modality globals or introduce shared/private branches.",
        "",
        f"Mistaken DSSA symbols present: `{code['forbidden_dssa_symbols_present']}`.",
    ]
    (audit_dir / "c16_mea_current_model_path.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_feasibility(audit_dir: Path, summary: Dict[str, Any]) -> str:
    checks = check_map(summary)
    bio = pd.read_csv(audit_dir / "c16_mea_available_bio_fields.csv")
    grouping = pd.read_csv(audit_dir / "c16_mea_bio_semantic_grouping.csv")
    temporal = pd.read_csv(audit_dir / "c16_mea_existing_temporal_fields.csv")
    c14 = pd.read_csv(audit_dir / "c16_mea_c14_diagnostic_mapping.csv")
    shortcuts = pd.read_csv(audit_dir / "c16_mea_shortcut_exclusion_map.csv")

    bio_grouping_valid = bool(summary["bio"]["all_bio_source_fields_present"])
    reference_ranges = bool(summary["bio"]["any_reference_range_available"])
    trusted_abnormal = int(summary["bio"]["trusted_abnormal_metadata_rows"])
    latest_rate = float(summary["temporal"].get("latest_focus_marker_fraction", 0.0))
    history_rate = float(summary["temporal"].get("history_focus_marker_fraction", 0.0))
    full_rate = float(summary["temporal"].get("full_report_marker_fraction", 0.0))
    audit_pass = bool(summary.get("audit_pass"))
    status = "C16_MEA_DESIGN_AUDIT_PASS_WITH_CONSTRAINTS" if audit_pass else "C16_MEA_DESIGN_AUDIT_FAIL"

    bio_path = (
        "Use verified `TgAb/TPOAb` as an immune-observed group, `FT3/FT4/TSH` as a thyroid-function-observed group, and `sex/age` as other observed context. Values remain continuous observed evidence with per-field validity masks."
        if bio_grouping_valid
        else "Use one neutral `bio_observed_evidence` node. Immune/function grouping is blocked until field names and order are repaired."
    )
    temporal_path = (
        "Build character-position masks from explicit C13 latest/history/full-report markers, and use learned fallback pooling when a section or role mask is absent."
        if latest_rate > 0 and full_rate > 0
        else "Keep the C13 report as one sequence and use learned full-report pooling only; do not fabricate visit boundaries."
    )

    check_frame = pd.DataFrame(
        [{"check": key, "pass": int(value)} for key, value in checks.items()]
    )
    module_frame = pd.DataFrame(
        [
            {
                "module": "ImageMorphologyEvidenceProjector",
                "verified_inputs": "per-image tokens + image_mask",
                "permitted_output": "learnable morphology-role slots and global fallback",
                "expected_auc_mechanism": "retain multiple morphology patterns instead of a single image mean",
                "constraint": "patient supervision only; slot names are architectural, not finding labels",
            },
            {
                "module": "TextEvidenceRoleProjector",
                "verified_inputs": "character tokens + report mask + audited dictionary/temporal position masks",
                "permitted_output": "support, opposition, uncertainty, nonspecific, temporal, global evidence",
                "expected_auc_mechanism": "separate coexisting support and opposition that C14 found visible but inconsistently used",
                "constraint": "masks guide pooling only; learned fallback for every empty mask; no weak-label BCE",
            },
            {
                "module": "BioEvidenceProjector",
                "verified_inputs": "seven ordered continuous values + per-field validity mask",
                "permitted_output": "immune-observed, thyroid-function-observed, and other-observed evidence",
                "expected_auc_mechanism": "preserve verified biochemical group structure without missingness counts",
                "constraint": "no abnormal/normal direction without reference ranges; role direction remains latent",
            },
            {
                "module": "HTMechanismRelationLayer",
                "verified_inputs": "audited modality evidence nodes",
                "permitted_output": "M1 morphology, M2 immune-observed, M3 function-observed, M4 opposition, M5 temporal, M6 disease state",
                "expected_auc_mechanism": "relate evidence through named HT mechanisms rather than raw-modality similarity",
                "constraint": "first alignment loss is image-text morphology only; no bio-text alignment without matching text semantics",
            },
            {
                "module": "EvidenceConflictAggregator",
                "verified_inputs": "support/opposition/uncertainty nodes + reliability masks",
                "permitted_output": "separate support, opposition, uncertainty, and conflict states",
                "expected_auc_mechanism": "avoid averaging contradictory evidence highlighted by C14",
                "constraint": "high conflict downweights alignment; modality availability is not a disease scalar",
            },
            {
                "module": "DiseaseStateAlignmentHead",
                "verified_inputs": "mechanism state + evidence states + conflict state",
                "permitted_output": "binary patient HT logit and diagnostics",
                "expected_auc_mechanism": "order patient support versus opposition while preserving ambiguous cases",
                "constraint": "binary task only; internal states are not new labels",
            },
        ]
    )
    loss_frame = pd.DataFrame(
        [
            {"loss": "L_cls", "weight": 1.0, "status": "required", "evidence_contract": "patient-level binary label"},
            {"loss": "L_state_margin", "weight": 0.03, "status": "allowed", "evidence_contract": "patient support-opposition ordering from training label"},
            {"loss": "L_mechanism_alignment", "weight": 0.02, "status": "allowed_with_scope", "evidence_contract": "image-text morphology only, valid-pair and low-conflict weighted"},
            {"loss": "L_role_separation", "weight": 0.005, "status": "allowed", "evidence_contract": "clinical support versus opposition states"},
            {"loss": "L_rank", "weight": 0.02, "status": "variant_B_only", "evidence_contract": "training-batch positive-negative pairs only"},
        ]
    )

    lines = [
        "# C16-MEA Design Feasibility",
        "",
        f"Design status: `{status}`.",
        "",
        "This report authorizes implementation only when every hard input check passes. It does not authorize training before static/synthetic validation and both seed-0 smoke gates.",
        "",
        "## Hard Checks",
        "",
        frame_to_markdown(check_frame),
        "",
        "## Bio Semantics",
        "",
        bio_path,
        "",
        f"Reference-range columns available: `{reference_ranges}`. Rows with trusted abnormal metadata: `{trusted_abnormal}`. Therefore no abnormal, normal, support, or opposition rule may be derived from bio values.",
        "",
        frame_to_markdown(
            bio,
            ["bio_index", "field_name", "semantic_group", "manifest_observed_fraction", "source_column_present", "reference_range_available", "abnormal_flag_one_count"],
        ),
        "",
        frame_to_markdown(grouping),
        "",
        "## Temporal Evidence",
        "",
        temporal_path,
        "",
        f"Latest marker coverage: `{latest_rate:.4f}`; history marker coverage: `{history_rate:.4f}`; full-report marker coverage: `{full_rate:.4f}`.",
        "",
        frame_to_markdown(temporal[temporal["item_type"] == "derived_temporal_state"]),
        "",
        "## C14 Evidence Basis",
        "",
        frame_to_markdown(c14, ["phase", "path", "available", "row_count", "purpose", "restriction", "route_or_status"]),
        "",
        "C14-A showed that relevant text evidence is generally exposed. C14-B found no single stable global modality-removal or fusion mechanism. C14-C/D localized many failures to hard patient subgroups, and C14-E failed the generalizability gate. C16-MEA therefore targets evidence organization and conflict handling, but it must not claim that C14 proves a general mechanism.",
        "",
        "## Proposed Modules",
        "",
        frame_to_markdown(module_frame),
        "",
        "## Loss Contract",
        "",
        frame_to_markdown(loss_frame),
        "",
        "Auxiliary weights use three classification-only warmup epochs and ramp through epoch 7. No broad sweep is allowed. Core and Core+Ranking are the only variants.",
        "",
        "## Shortcut Exclusion",
        "",
        frame_to_markdown(shortcuts, ["field", "item_type", "allowed_as_model_input", "allowed_as_loss_or_gate_input", "implementation_rule"]),
        "",
        "## Explicitly Blocked",
        "",
        "- No shared/private, modality-invariant, DecAlign, or generic orthogonality modules.",
        "- No report-derived image labels or old morphology BCE losses.",
        "- No rule-based bio abnormality, reference range, or support/opposition target.",
        "- No immune/function cross-modal alignment unless matching text semantics are separately verified.",
        "- No shortcut counts, source folder, report length, modality-presence scalar, or C14 hard-group field in the predictor.",
        "- No test-based architecture, variant, checkpoint, threshold, loss, or route selection.",
        "",
        "## Exact Expected AUC Mechanism",
        "",
        "The proposed change is intended to improve patient ranking by keeping visible supporting and opposing evidence separate, representing temporal contradiction explicitly, and relating only clinically compatible evidence through named mechanism nodes. This addresses the observed failure of evidence use and pairwise ordering without assuming that one modality should be globally removed or that raw modalities should share a representation.",
        "",
        "## Next Gate",
        "",
        "If this design status passes, implementation may begin with backward-compatible modules and static/synthetic checks. Training remains blocked until both Core and Core+Ranking seed-0 smoke configurations pass all collapse, saturation, compatibility, and shortcut checks.",
    ]
    (audit_dir / "c16_mea_design_feasibility.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return status


def collect(args: argparse.Namespace) -> Dict[str, Any]:
    audit_dir = Path(args.audit_dir)
    summary_path = audit_dir / "c16_mea_design_audit_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    write_current_model_path(audit_dir, summary)
    status = write_feasibility(audit_dir, summary)
    inventory = []
    for name in REQUIRED_FILES:
        path = audit_dir / name
        inventory.append({"file": name, "exists": int(path.is_file()), "bytes": path.stat().st_size if path.is_file() else 0})
    inventory_frame = pd.DataFrame(inventory)
    inventory_frame.to_csv(audit_dir / "c16_mea_design_file_inventory.csv", index=False)
    delivery_pass = bool((inventory_frame["exists"] == 1).all() and summary.get("audit_pass"))
    result = {
        "design_status": status,
        "audit_pass": bool(summary.get("audit_pass")),
        "delivery_pass": delivery_pass,
        "required_files": len(REQUIRED_FILES),
        "output_dir": str(audit_dir),
    }
    (audit_dir / "c16_mea_design_gate.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if args.require_pass and not delivery_pass:
        raise SystemExit(1)
    return result


def main() -> None:
    args = parse_args()
    print(json.dumps(collect(args), ensure_ascii=False))


if __name__ == "__main__":
    main()
