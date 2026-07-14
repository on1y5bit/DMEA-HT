#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dmea_ht.visit_data import (  # noqa: E402
    IMAGE_SUFFIXES,
    group_selected_images,
    load_source_visits,
    normalize_date,
    normalize_patient_id,
    read_jsonl,
    sha256_file,
)


def report_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest() if text else ""


def label_counts(rows: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    counts = collections.Counter((str(row["split"]), int(row["label"])) for row in rows)
    return {f"{split}_{label}": int(count) for (split, label), count in sorted(counts.items())}


def historical_source_dates(data_root: Path, label: int, patient_id: str) -> List[str]:
    patient_dir = data_root / str(label) / patient_id
    dates: List[str] = []
    if not patient_dir.exists():
        return dates
    for visit_dir in patient_dir.iterdir():
        if not visit_dir.is_dir():
            continue
        has_image = any(
            path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            for path in visit_dir.rglob("*")
        )
        if has_image:
            dates.append(normalize_date(visit_dir.name))
    dates = sorted(date for date in dates if date)
    if len(dates) <= 1:
        return dates
    final_year = max(date[:4] for date in dates)
    return [date for date in dates if date[:4] < final_year]


def metric_row(name: str, value: Any, threshold: str, passed: bool) -> Dict[str, Any]:
    return {"metric": name, "value": value, "threshold": threshold, "pass": bool(passed)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit C27 visit reconstruction against real source data.")
    parser.add_argument("--data-root", default="/data/csb/DMEA-HT/HT_2025.12_25")
    parser.add_argument(
        "--base-manifest",
        default="/data/csb/DMEA-HT/HT_2025.12_25/manifest_distmatch_structmatch_evidence_v2_c13_temporal_focus.jsonl",
    )
    parser.add_argument(
        "--visit-manifest",
        default="/data/csb/DMEA-HT/HT_2025.12_25/manifest_distmatch_structmatch_evidence_v2_c27_visit_level.jsonl",
    )
    parser.add_argument("--output-dir", default="analysis_reports/phase_c27_visit_design")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    base_rows = read_jsonl(args.base_manifest)
    visit_rows = read_jsonl(args.visit_manifest)
    source_visits = load_source_visits(data_root)
    base_by_patient = {normalize_patient_id(row["patient_id"]): row for row in base_rows}
    visit_by_patient = {normalize_patient_id(row["patient_id"]): row for row in visit_rows}

    patient_rows: List[Dict[str, Any]] = []
    reconstructed_visit_rows: List[Dict[str, Any]] = []
    image_owners: Dict[str, set[Tuple[str, str]]] = collections.defaultdict(set)
    original_field_changes = 0
    report_source_mismatches = 0
    bio_source_mismatches = 0
    missing_report_copies = 0
    visit_order_failures = 0
    history_cutoff_failures = 0
    selected_visit_count_failures = 0
    image_grouping_failures = 0
    missing_image_files = 0

    for patient_id, base in base_by_patient.items():
        rebuilt = visit_by_patient.get(patient_id)
        if rebuilt is None:
            patient_rows.append({"patient_id": patient_id, "missing_from_visit_manifest": True})
            continue
        for key, value in base.items():
            if rebuilt.get(key) != value:
                original_field_changes += 1

        selected_dates = [normalize_date(value) for value in base.get("selected_visit_dates", [])]
        expected_dates = sorted(selected_dates)
        visits = list(rebuilt.get("visits", []))
        rebuilt_dates = [normalize_date(visit.get("visit_date")) for visit in visits]
        ranks = [int(visit.get("visit_rank", -1)) for visit in visits]
        order_ok = rebuilt_dates == expected_dates and ranks == list(range(len(visits)))
        if not order_ok:
            visit_order_failures += 1
        count_ok = len(visits) == len(selected_dates) == int(base.get("selected_n_visits", len(selected_dates)))
        if not count_ok:
            selected_visit_count_failures += 1

        eligible_dates = historical_source_dates(data_root, int(base["label"]), patient_id)
        history_ok = set(rebuilt_dates).issubset(set(eligible_dates))
        if not history_ok:
            history_cutoff_failures += 1
        expected_images = group_selected_images(base)
        patient_image_grouping_ok = True
        report_count = 0
        dated_bio_count = 0
        for visit in visits:
            visit_date = normalize_date(visit["visit_date"])
            source = source_visits.get((patient_id, visit_date))
            image_paths = [str(path).replace("\\", "/") for path in visit.get("image_paths", [])]
            image_group_ok = image_paths == expected_images.get(visit_date, [])
            patient_image_grouping_ok = patient_image_grouping_ok and image_group_ok
            if not image_group_ok:
                image_grouping_failures += 1
            for path in image_paths:
                image_owners[path].add((patient_id, visit_date))
                if not (data_root / path).is_file():
                    missing_image_files += 1

            report_text = str(visit.get("report_text", ""))
            expected_report = source.report_text if source is not None else ""
            report_ok = report_text == expected_report
            if not report_ok:
                report_source_mismatches += 1
            if source is None and report_text:
                missing_report_copies += 1
            has_report = bool(report_text.strip())
            report_count += int(has_report)

            expected_bio_values = list(source.bio_values) if source is not None else [0.0] * 7
            expected_bio_missing = list(source.bio_missing_mask) if source is not None else [1] * 7
            bio_ok = (
                list(visit.get("bio_values_if_dated", [])) == expected_bio_values
                and list(visit.get("bio_missing_mask_if_dated", [])) == expected_bio_missing
                and visit.get("dated_bio_row_id") == (source.bio_source_row_id if source is not None else None)
            )
            if not bio_ok:
                bio_source_mismatches += 1
            has_dated_bio = source is not None
            dated_bio_count += int(has_dated_bio)
            reconstructed_visit_rows.append(
                {
                    "patient_id": patient_id,
                    "split": base["split"],
                    "label": int(base["label"]),
                    "visit_id": visit.get("visit_id", ""),
                    "visit_date": visit_date,
                    "visit_rank_oldest_to_latest": int(visit.get("visit_rank", -1)),
                    "image_paths_for_visit": "|".join(image_paths),
                    "n_image_entries": len(image_paths),
                    "all_image_paths_exist": all((data_root / path).is_file() for path in image_paths),
                    "visit_report_present": has_report,
                    "visit_report_length": len(report_text),
                    "visit_report_sha256": report_sha256(report_text),
                    "visit_report_source": visit.get("report_source", ""),
                    "visit_report_source_matches": report_ok,
                    "dated_bio_row_id": visit.get("dated_bio_row_id"),
                    "bio_time_relation": visit.get("bio_time_relation", ""),
                    "dated_bio_present": has_dated_bio,
                    "dated_bio_source_matches": bio_ok,
                    "history_cutoff_eligible": visit_date in eligible_dates,
                }
            )

        patient_rows.append(
            {
                "patient_id": patient_id,
                "split": base["split"],
                "label": int(base["label"]),
                "selected_visit_count": len(selected_dates),
                "reconstructed_visit_count": len(visits),
                "reconstructable_report_count": report_count,
                "visit_report_coverage": report_count / max(len(visits), 1),
                "dated_bio_visit_count": dated_bio_count,
                "date_order_reproducible": order_ok,
                "selected_visit_count_matches": count_ok,
                "image_grouping_matches": patient_image_grouping_ok,
                "history_cutoff_matches_c13": history_ok,
                "patient_bio_fallback_used_once": bool(rebuilt.get("patient_bio_fallback", {}).get("valid", False)),
                "missing_from_visit_manifest": False,
            }
        )

    patient_frame = pd.DataFrame(patient_rows).sort_values("patient_id").reset_index(drop=True)
    visit_frame = pd.DataFrame(reconstructed_visit_rows).sort_values(
        ["patient_id", "visit_rank_oldest_to_latest"]
    ).reset_index(drop=True)
    cross_patient_image_leakage = sum(len(owners) > 1 for owners in image_owners.values())
    selected_visits = len(visit_frame)
    visits_with_image = int((visit_frame["n_image_entries"] > 0).sum())
    visits_with_report = int(visit_frame["visit_report_present"].sum())
    visits_with_both = int(((visit_frame["n_image_entries"] > 0) & visit_frame["visit_report_present"]).sum())
    multi_patient = patient_frame[patient_frame["selected_visit_count"] > 1]
    multi_val = multi_patient[multi_patient["split"].eq("val")]
    multi_val_two_report = int((multi_val["reconstructable_report_count"] >= 2).sum())
    visit_report_coverage = visits_with_report / max(selected_visits, 1)
    multi_val_two_report_coverage = multi_val_two_report / max(len(multi_val), 1)
    source_duplicate_patient_dates = sum(len(source.source_row_ids) > 1 for source in source_visits.values())

    expected_counts = {"train_0": 301, "train_1": 301, "val_0": 47, "val_1": 47, "test_0": 42, "test_1": 42}
    actual_counts = label_counts(visit_rows)
    patient_set_ok = set(base_by_patient) == set(visit_by_patient) and len(visit_rows) == len(base_rows) == 780
    split_label_ok = actual_counts == expected_counts == label_counts(base_rows)
    hard_checks = {
        "patient_set_and_count": patient_set_ok,
        "patient_ids_unique": len(visit_by_patient) == len(visit_rows),
        "split_label_invariance": split_label_ok,
        "original_patient_fields_unchanged": original_field_changes == 0,
        "selected_visit_counts_match": selected_visit_count_failures == 0,
        "visit_order_reproducible": visit_order_failures == 0,
        "history_cutoff_matches_c13": history_cutoff_failures == 0,
        "image_grouping_matches_c13": image_grouping_failures == 0,
        "all_selected_image_paths_exist": missing_image_files == 0,
        "cross_patient_image_leakage_zero": cross_patient_image_leakage == 0,
        "visit_reports_match_real_source": report_source_mismatches == 0,
        "missing_reports_not_filled_from_patient_text": missing_report_copies == 0,
        "dated_bio_matches_exact_source_date": bio_source_mismatches == 0,
        "source_patient_dates_unique": source_duplicate_patient_dates == 0,
        "test_not_used_for_rule_design": all(
            not bool(row.get("c27_visit_reconstruction", {}).get("test_used_for_rule_design", True))
            for row in visit_rows
        ),
    }
    hard_pass = all(hard_checks.values())
    coverage_rows = [
        metric_row("patients", len(visit_rows), "==780", len(visit_rows) == 780),
        metric_row("total_selected_visits", selected_visits, ">0", selected_visits > 0),
        metric_row("visits_with_dated_image", visits_with_image, f"=={selected_visits}", visits_with_image == selected_visits),
        metric_row("visits_with_visit_level_report_text", visits_with_report, "audit", True),
        metric_row("visits_with_both_image_and_text", visits_with_both, "audit", True),
        metric_row("visit_report_reconstruction_coverage", visit_report_coverage, ">=0.80", visit_report_coverage >= 0.80),
        metric_row("multi_visit_patients", len(multi_patient), "audit", True),
        metric_row("multi_visit_validation_patients", len(multi_val), "audit", True),
        metric_row("multi_visit_validation_patients_with_two_report_blocks", multi_val_two_report, "audit", True),
        metric_row("multi_visit_validation_two_block_coverage", multi_val_two_report_coverage, ">=0.70", multi_val_two_report_coverage >= 0.70),
        metric_row("cross_patient_image_leakage_count", cross_patient_image_leakage, "==0", cross_patient_image_leakage == 0),
        metric_row("report_source_mismatch_count", report_source_mismatches, "==0", report_source_mismatches == 0),
        metric_row("bio_source_mismatch_count", bio_source_mismatches, "==0", bio_source_mismatches == 0),
    ]

    patient_frame.to_csv(output / "c27_visit_reconstruction_patient_audit.csv", index=False)
    visit_frame.to_csv(output / "c27_visit_reconstruction_visit_audit.csv", index=False)
    pd.DataFrame(coverage_rows).to_csv(output / "c27_visit_reconstruction_coverage.csv", index=False)
    invariance = {
        "phase": "C27-VTME",
        "base_manifest": str(Path(args.base_manifest).resolve()),
        "base_manifest_sha256": sha256_file(args.base_manifest),
        "visit_manifest": str(Path(args.visit_manifest).resolve()),
        "visit_manifest_sha256": sha256_file(args.visit_manifest),
        "patients": len(visit_rows),
        "selected_visits": selected_visits,
        "base_split_label_counts": label_counts(base_rows),
        "visit_split_label_counts": actual_counts,
        "hard_checks": hard_checks,
        "hard_pass": hard_pass,
        "visit_report_coverage": visit_report_coverage,
        "multi_visit_validation_two_block_coverage": multi_val_two_report_coverage,
        "test_used_for_rule_design": False,
    }
    (output / "c27_visit_manifest_invariance.json").write_text(
        json.dumps(invariance, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(invariance))


if __name__ == "__main__":
    main()
