#!/usr/bin/env python3
"""Strict C66 fold-local protocol utilities.

This module deliberately reconstructs the C66 development protocol from the
saved C64 development-only artifacts.  It never opens the full manifest, so
the locked Test records are not parsed during C66-A.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(REPO_ROOT))

from dmea_ht.config import load_config  # noqa: E402


SEEDS = (0, 42, 3407)
FOLD_COUNT = 5
FOLD_SEED = 20260716
DEVELOPMENT_PATIENT_COUNT = 696


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_c66_config(path: str | Path) -> Dict[str, Any]:
    config = load_config(resolve_path(path))
    if str(config.get("phase", "")).lower() != "c66":
        raise RuntimeError("C66 configuration phase mismatch")
    if tuple(int(seed) for seed in config.get("formal_seeds", ())) != SEEDS:
        raise RuntimeError("C66 formal seeds must remain [0, 42, 3407]")

    outer = dict(config.get("outer_folds", {}))
    if int(outer.get("count", -1)) != FOLD_COUNT or int(outer.get("seed", -1)) != FOLD_SEED:
        raise RuntimeError("C66 must reuse the C64 five-fold seed 20260716")
    if [int(value) for value in outer.get("expected_sizes", [])] != [140, 140, 140, 138, 138]:
        raise RuntimeError("C66 outer-fold sizes must remain 140/140/140/138/138")

    inner = dict(config.get("inner_validation", {}))
    if float(inner.get("fraction", -1.0)) != 0.20:
        raise RuntimeError("C66 inner validation fraction must remain 0.20")
    if int(inner.get("seed_base", -1)) != 20260718:
        raise RuntimeError("C66 inner split seed base must remain 20260718")
    if not bool(inner.get("patient_level")) or not bool(inner.get("stratified_by_label")):
        raise RuntimeError("C66 inner splits must be patient-level and label-stratified")

    data_contract = dict(config.get("data_contract", {}))
    if int(data_contract.get("development_patient_count", -1)) != DEVELOPMENT_PATIENT_COUNT:
        raise RuntimeError("C66 development pool must contain 696 patients")
    if bool(data_contract.get("test_reading_before_final_gate", True)):
        raise RuntimeError("C66 must lock Test before the final gate")

    initialization = dict(config.get("initialization_contract", {}))
    if initialization.get("task_checkpoint_initialization") != "forbidden":
        raise RuntimeError("C66 must forbid task checkpoint initialization")
    if initialization.get("historical_prediction_input") != "forbidden":
        raise RuntimeError("C66 must forbid historical prediction inputs")
    if initialization.get("historical_representation_input") != "forbidden":
        raise RuntimeError("C66 must forbid historical representation inputs")

    for stage in ("source_learning", "route_training"):
        section = dict(config.get(stage, {}))
        if int(section.get("max_epochs", -1)) != 60 or int(section.get("patience", -1)) != 15:
            raise RuntimeError(f"C66 {stage} must retain max_epochs=60 and patience=15")
    return config


def report_dir(config: Mapping[str, Any]) -> Path:
    return resolve_path(config["project"]["report_dir"])


def nested_cv_dir(config: Mapping[str, Any]) -> Path:
    return resolve_path(config["project"]["nested_cv_output_dir"])


def c64_inventory_path(config: Mapping[str, Any]) -> Path:
    return resolve_path(config["project"]["c64_report_dir"]) / "c64_fold_patient_inventory.csv"


def c64_assignments_path(config: Mapping[str, Any]) -> Path:
    return resolve_path(config["project"]["c64_cv_output_dir"]) / "fold_assignments.json"


def read_c64_development_inventory(config: Mapping[str, Any]) -> pd.DataFrame:
    """Read only C64's development-only patient inventory, never the manifest."""
    path = c64_inventory_path(config)
    if not path.exists():
        raise FileNotFoundError(
            f"C66-A cannot reconstruct the original C61 source without C64 inventory: {path}"
        )
    frame = pd.read_csv(path, dtype={"patient_id": str})
    required = {"patient_id", "label", "original_split", "fold"}
    missing = required.difference(frame.columns)
    if missing:
        raise RuntimeError(f"C64 inventory is missing columns: {sorted(missing)}")
    frame = frame.loc[:, ["patient_id", "label", "original_split", "fold"]].copy()
    frame["patient_id"] = frame["patient_id"].astype(str)
    frame["label"] = frame["label"].astype(int)
    frame["original_split"] = frame["original_split"].astype(str).str.lower()
    frame["fold"] = frame["fold"].astype(int)

    if len(frame) != DEVELOPMENT_PATIENT_COUNT or frame["patient_id"].nunique() != DEVELOPMENT_PATIENT_COUNT:
        raise RuntimeError("C64 development inventory must contain exactly 696 unique patients")
    if set(frame["label"].unique()) != {0, 1}:
        raise RuntimeError("C64 development inventory must retain both labels")
    if not set(frame["original_split"].unique()).issubset({"train", "val"}):
        raise RuntimeError("C64 development inventory unexpectedly contains a non-development split")
    if int((frame["original_split"] == "train").sum()) != 602:
        raise RuntimeError("C66-A expected exactly 602 original C61 Train patients")
    if int((frame["original_split"] == "val").sum()) != 94:
        raise RuntimeError("C66-A expected exactly 94 original C61 Validation patients")
    if sorted(frame.groupby("fold").size().tolist()) != [138, 138, 140, 140, 140]:
        raise RuntimeError("C64 inventory fold sizes do not match the frozen C64 partition")
    return frame.sort_values("patient_id").reset_index(drop=True)


def load_c64_outer_assignments(config: Mapping[str, Any], inventory: pd.DataFrame) -> Dict[int, list[str]]:
    path = c64_assignments_path(config)
    if not path.exists():
        raise FileNotFoundError(f"C66-A requires the frozen C64 fold assignment artifact: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if set(payload) != {str(index) for index in range(FOLD_COUNT)}:
        raise RuntimeError("C64 fold assignment must contain exactly folds 0 through 4")

    development_ids = set(inventory["patient_id"].tolist())
    observed: list[str] = []
    assignments: Dict[int, list[str]] = {}
    for fold in range(FOLD_COUNT):
        patient_ids = [str(patient_id) for patient_id in payload[str(fold)]]
        if len(patient_ids) != len(set(patient_ids)):
            raise RuntimeError(f"C64 fold {fold} contains duplicate patient IDs")
        assignments[fold] = sorted(patient_ids)
        observed.extend(patient_ids)
    if len(observed) != DEVELOPMENT_PATIENT_COUNT or set(observed) != development_ids:
        raise RuntimeError("C64 fold assignments do not exactly cover the 696-patient development pool")
    if len(observed) != len(set(observed)):
        raise RuntimeError("C64 fold assignments overlap across outer folds")

    inventory_assignments = {
        int(fold): set(group["patient_id"].tolist()) for fold, group in inventory.groupby("fold")
    }
    for fold, patient_ids in assignments.items():
        if set(patient_ids) != inventory_assignments.get(fold, set()):
            raise RuntimeError(f"C64 JSON and inventory disagree for outer fold {fold}")
    return assignments


def stratified_inner_split(
    inventory: pd.DataFrame, outer_train_ids: Iterable[str], seed: int, fraction: float
) -> tuple[list[str], list[str]]:
    outer_train = set(str(patient_id) for patient_id in outer_train_ids)
    subset = inventory[inventory["patient_id"].isin(outer_train)]
    if len(subset) != len(outer_train):
        raise RuntimeError("Inner split received an unknown patient ID")

    rng = random.Random(int(seed))
    inner_val: list[str] = []
    for label in (0, 1):
        patient_ids = sorted(subset.loc[subset["label"] == label, "patient_id"].astype(str).tolist())
        if len(patient_ids) < 2:
            raise RuntimeError("C66 inner stratification requires at least two patients per label")
        rng.shuffle(patient_ids)
        count = max(1, min(len(patient_ids) - 1, int(round(len(patient_ids) * float(fraction)))))
        inner_val.extend(patient_ids[:count])
    inner_val = sorted(inner_val)
    inner_train = sorted(outer_train.difference(inner_val))
    if not inner_train or not inner_val:
        raise RuntimeError("C66 inner split produced an empty partition")
    return inner_train, inner_val


def build_nested_split_payload(config: Mapping[str, Any]) -> Dict[str, Any]:
    inventory = read_c64_development_inventory(config)
    assignments = load_c64_outer_assignments(config, inventory)
    development_ids = set(inventory["patient_id"].tolist())
    fraction = float(config["inner_validation"]["fraction"])
    seed_base = int(config["inner_validation"]["seed_base"])

    folds: Dict[str, Dict[str, Any]] = {}
    for fold in range(FOLD_COUNT):
        outer_val = assignments[fold]
        outer_train = sorted(development_ids.difference(outer_val))
        inner_seed = seed_base + fold
        inner_train, inner_val = stratified_inner_split(inventory, outer_train, inner_seed, fraction)
        folds[str(fold)] = {
            "outer_fold": fold,
            "inner_split_seed": inner_seed,
            "outer_train_patient_ids": outer_train,
            "outer_val_patient_ids": outer_val,
            "inner_train_patient_ids": inner_train,
            "inner_val_patient_ids": inner_val,
        }

    return {
        "phase": "C66-LFFC",
        "stage": "C66-A nested split construction",
        "outer_fold_seed": FOLD_SEED,
        "inner_validation_fraction": fraction,
        "development_patient_count": DEVELOPMENT_PATIENT_COUNT,
        "test_loaded": False,
        "test_rows_read": 0,
        "manifest_opened": False,
        "source_artifacts": {
            "c64_development_inventory": str(c64_inventory_path(config)),
            "c64_fold_assignments": str(c64_assignments_path(config)),
        },
        "folds": folds,
    }


def nested_split_path(config: Mapping[str, Any]) -> Path:
    return nested_cv_dir(config) / "splits" / "nested_split_assignments.json"


def nested_split_summary(payload: Mapping[str, Any], inventory: pd.DataFrame) -> pd.DataFrame:
    labels = inventory.set_index("patient_id")["label"].to_dict()
    rows = []
    for key, fold_payload in sorted(dict(payload["folds"]).items(), key=lambda item: int(item[0])):
        for split_name, field in (
            ("outer_train", "outer_train_patient_ids"),
            ("outer_val", "outer_val_patient_ids"),
            ("inner_train", "inner_train_patient_ids"),
            ("inner_val", "inner_val_patient_ids"),
        ):
            patient_ids = [str(patient_id) for patient_id in fold_payload[field]]
            rows.append(
                {
                    "outer_fold": int(key),
                    "split": split_name,
                    "patient_count": len(patient_ids),
                    "label0_count": sum(labels[patient_id] == 0 for patient_id in patient_ids),
                    "label1_count": sum(labels[patient_id] == 1 for patient_id in patient_ids),
                    "inner_split_seed": int(fold_payload["inner_split_seed"]),
                }
            )
    return pd.DataFrame(rows).sort_values(["outer_fold", "split"]).reset_index(drop=True)


def validate_nested_split_payload(
    payload: Mapping[str, Any], inventory: pd.DataFrame, config: Mapping[str, Any]
) -> Dict[str, Any]:
    development_ids = set(inventory["patient_id"].astype(str).tolist())
    expected_sizes = [int(value) for value in config["outer_folds"]["expected_sizes"]]
    checks: list[Dict[str, Any]] = []
    fold_audits: Dict[str, Dict[str, Any]] = {}
    folds = dict(payload.get("folds", {}))
    if set(folds) != {str(index) for index in range(FOLD_COUNT)}:
        checks.append({"name": "five_outer_folds_present", "passed": False, "detail": sorted(folds)})
        return {"all_pass": False, "checks": checks, "folds": fold_audits}

    all_outer_val: list[str] = []
    for fold in range(FOLD_COUNT):
        entry = dict(folds[str(fold)])
        outer_train = set(map(str, entry.get("outer_train_patient_ids", [])))
        outer_val = set(map(str, entry.get("outer_val_patient_ids", [])))
        inner_train = set(map(str, entry.get("inner_train_patient_ids", [])))
        inner_val = set(map(str, entry.get("inner_val_patient_ids", [])))
        expected_inner_seed = int(config["inner_validation"]["seed_base"]) + fold

        fold_checks = {
            "outer_val_size": len(outer_val) == expected_sizes[fold],
            "outer_partition_complete": outer_train | outer_val == development_ids,
            "outer_partition_disjoint": not bool(outer_train & outer_val),
            "inner_partition_complete": inner_train | inner_val == outer_train,
            "inner_partition_disjoint": not bool(inner_train & inner_val),
            "outer_val_not_in_source_training": not bool(outer_val & inner_train),
            "outer_val_not_in_inner_validation": not bool(outer_val & inner_val),
            "outer_val_not_in_route_selection": not bool(outer_val & inner_val),
            "outer_val_not_in_epoch_selection": not bool(outer_val & inner_val),
            "outer_val_not_in_outer_refit": not bool(outer_val & outer_train),
            "inner_seed_matches_contract": int(entry.get("inner_split_seed", -1)) == expected_inner_seed,
            "all_ids_are_development_ids": (outer_train | outer_val | inner_train | inner_val).issubset(development_ids),
        }
        fold_audits[str(fold)] = {
            "outer_train_count": len(outer_train),
            "outer_val_count": len(outer_val),
            "inner_train_count": len(inner_train),
            "inner_val_count": len(inner_val),
            "outer_val_overlap_source_training": len(outer_val & inner_train),
            "outer_val_overlap_inner_validation": len(outer_val & inner_val),
            "outer_val_overlap_outer_refit": len(outer_val & outer_train),
            "checks": fold_checks,
            "all_pass": all(fold_checks.values()),
        }
        all_outer_val.extend(sorted(outer_val))

    checks.extend(
        [
            {
                "name": "outer_validation_covers_each_development_patient_once",
                "passed": len(all_outer_val) == DEVELOPMENT_PATIENT_COUNT
                and len(set(all_outer_val)) == DEVELOPMENT_PATIENT_COUNT
                and set(all_outer_val) == development_ids,
                "detail": {"outer_val_rows": len(all_outer_val), "unique_outer_val_patients": len(set(all_outer_val))},
            },
            {"name": "test_not_loaded", "passed": payload.get("test_loaded") is False, "detail": payload.get("test_loaded")},
            {"name": "test_rows_not_read", "passed": int(payload.get("test_rows_read", -1)) == 0, "detail": payload.get("test_rows_read")},
            {"name": "manifest_not_opened", "passed": payload.get("manifest_opened") is False, "detail": payload.get("manifest_opened")},
        ]
    )
    all_pass = all(check["passed"] for check in checks) and all(audit["all_pass"] for audit in fold_audits.values())
    return {"all_pass": all_pass, "checks": checks, "folds": fold_audits}
