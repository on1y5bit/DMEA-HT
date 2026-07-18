#!/usr/bin/env python3
"""Audit C66 prior C61 exposure and fold-local nested-CV isolation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts import c66_common as common  # noqa: E402


def initialization_inventory(config: Mapping[str, Any]) -> pd.DataFrame:
    contract = dict(config["initialization_contract"])
    rows = []
    for module, required in (
        ("image_encoder", dict(contract["image_encoder"])),
        ("text_encoder", dict(contract["text_encoder"])),
        ("bio_source_encoder", {"required_initialization": "deterministic_seed_random"}),
        ("evidence_projectors", {"required_initialization": "deterministic_seed_random"}),
        ("source_evidence_stack", {"required_initialization": "deterministic_seed_random"}),
        ("temporary_source_classifier", {"required_initialization": "deterministic_seed_random"}),
        ("cbpi_task_path", {"required_initialization": "deterministic_seed_random"}),
    ):
        rows.append(
            {
                "module": module,
                "required_initialization": required["required_initialization"],
                "analysis_state": "not_instantiated_analysis_only",
                "public_source": "runtime_preflight_required" if module in {"image_encoder", "text_encoder"} else "not_applicable",
                "public_version": "runtime_preflight_required" if module in {"image_encoder", "text_encoder"} else "not_applicable",
                "public_sha256": "runtime_preflight_required" if module in {"image_encoder", "text_encoder"} else "not_applicable",
                "task_checkpoint_used": False,
                "prohibited_task_checkpoint_range": contract["prohibited_task_checkpoint_range"],
                "historical_prediction_input": False,
                "historical_representation_input": False,
                "training_permitted_from_this_row": False,
            }
        )
    return pd.DataFrame(rows)


def overlap_table(
    inventory: pd.DataFrame, payload: Mapping[str, Any]
) -> pd.DataFrame:
    original_split = inventory.set_index("patient_id")["original_split"].to_dict()
    labels = inventory.set_index("patient_id")["label"].to_dict()
    rows = []
    for fold, entry in sorted(dict(payload["folds"]).items(), key=lambda item: int(item[0])):
        outer_val = [str(patient_id) for patient_id in entry["outer_val_patient_ids"]]
        train_count = sum(original_split[patient_id] == "train" for patient_id in outer_val)
        validation_count = sum(original_split[patient_id] == "val" for patient_id in outer_val)
        never_count = len(outer_val) - train_count - validation_count
        rows.append(
            {
                "outer_fold": int(fold),
                "outer_val_patient_count": len(outer_val),
                "outer_val_label0_count": sum(labels[patient_id] == 0 for patient_id in outer_val),
                "outer_val_label1_count": sum(labels[patient_id] == 1 for patient_id in outer_val),
                "prior_c61_train_patient_count": train_count,
                "prior_c61_validation_patient_count": validation_count,
                "never_used_by_c61_patient_count": never_count,
                "prior_c61_train_fraction": train_count / max(len(outer_val), 1),
                "prior_c61_validation_fraction": validation_count / max(len(outer_val), 1),
            }
        )
    return pd.DataFrame(rows).sort_values("outer_fold").reset_index(drop=True)


def provisional_decision(report_dir: Path, overlap: pd.DataFrame, integrity: Mapping[str, Any]) -> None:
    lines = [
        "# C66-A Leakage-Free Nested-CV Audit",
        "",
        "Status: `PENDING_C66A_GATE`",
        "",
        "C66-A used only C64's development-only inventory and frozen fold-assignment artifacts. The full manifest and locked Test records were not opened.",
        "",
        "## Prior C61 Exposure",
        "",
        f"- Outer-validation patients previously in C61 Train: `{int(overlap['prior_c61_train_patient_count'].sum())}` across the five folds.",
        f"- Outer-validation patients previously in C61 Validation: `{int(overlap['prior_c61_validation_patient_count'].sum())}` across the five folds.",
        f"- Outer-validation patients never used by C61: `{int(overlap['never_used_by_c61_patient_count'].sum())}` across the five folds.",
        "- This confirms why C64/C65 are not interpreted as strict leakage-free OOF estimates.",
        "",
        "## C66 Isolation",
        "",
        f"- Nested split isolation checks passed before the gate: `{bool(integrity['all_pass'])}`.",
        "- Every C66 outer-validation patient is excluded from fold-local source training, inner validation, route selection, epoch selection, and outer-train refitting.",
        "- This is an analysis-only artifact. Public generic image/text initialization must still be instantiated with source, version, and SHA256 recorded by the future source-training preflight; no training is authorized by this provisional file alone.",
        "",
    ]
    (report_dir / "c66a_route_decision.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dema_ht_c66_source_learning.yaml")
    args = parser.parse_args()

    config = common.load_c66_config(args.config)
    report_dir = common.report_dir(config)
    split_path = common.nested_split_path(config)
    if not split_path.exists():
        raise FileNotFoundError(f"C66-A nested split artifact is missing: {split_path}")
    payload = json.loads(split_path.read_text(encoding="utf-8"))
    inventory = common.read_c64_development_inventory(config)
    integrity = common.validate_nested_split_payload(payload, inventory, config)
    overlap = overlap_table(inventory, payload)
    init_inventory = initialization_inventory(config)

    report_dir.mkdir(parents=True, exist_ok=True)
    overlap.to_csv(report_dir / "c66a_prior_checkpoint_overlap_by_fold.csv", index=False)
    init_inventory.to_csv(report_dir / "c66a_initialization_inventory.csv", index=False)
    audit: Dict[str, Any] = {
        "phase": "C66-LFFC",
        "stage": "C66-A prior checkpoint overlap audit",
        "status": "C66A_AUDIT_COMPLETE" if integrity["all_pass"] else "C66A_PROTOCOL_RECONSTRUCTION_FAIL",
        "prior_c61_exposure_reconstructed": bool(
            int(overlap["prior_c61_train_patient_count"].sum()) == 602
            and int(overlap["prior_c61_validation_patient_count"].sum()) == 94
            and int(overlap["never_used_by_c61_patient_count"].sum()) == 0
        ),
        "prior_c61_exposure_totals": {
            "train": int(overlap["prior_c61_train_patient_count"].sum()),
            "validation": int(overlap["prior_c61_validation_patient_count"].sum()),
            "never_used": int(overlap["never_used_by_c61_patient_count"].sum()),
        },
        "nested_cv_isolation": integrity,
        "test_loaded": False,
        "test_rows_read": 0,
        "manifest_opened": False,
        "task_checkpoint_loaded": False,
        "historical_prediction_or_representation_input": False,
        "source_training_authorized_by_this_audit": False,
    }
    common.write_json(report_dir / "c66a_prior_cv_leakage_audit.json", audit)
    provisional_decision(report_dir, overlap, integrity)
    print(json.dumps({"phase": "C66-LFFC", "status": audit["status"], "test_loaded": False}))


if __name__ == "__main__":
    main()
