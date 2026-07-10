from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analyze_phase_c11_report_filter_hypotheses import (  # noqa: E402
    DIFFUSE_HT_CUES,
    MORPHOLOGY_CUES,
    NEGATIVE_THYROID_CUES,
    NON_THYROID_CUES,
    THYROID_CUES,
    count_terms,
    parse_visits,
    split_clauses,
)


BENIGN_NODULE_CUES = [
    "结节",
    "低回声结节",
    "无回声",
    "边界清",
    "边界清晰",
    "形态规则",
    "椭圆",
    "囊",
    "后方回声无明显改变",
    "内部回声均匀",
]

PATIENT_OUTPUT_COLUMNS = [
    "patient_id",
    "split",
    "label",
    "seed",
    "pred_prob",
    "pred_label",
    "confusion_type",
    "case_type",
    "cross_seed_fn_count",
    "cross_seed_tp_count",
    "cross_seed_pred_mean",
    "cross_seed_pred_std",
    "seed_sensitive",
    "report_length_chars",
    "report_length_tokens_if_available",
    "audit_window_type",
    "model_text_char_window",
    "selected_n_visits",
    "used_images",
    "has_bio",
    "bio_missing_count",
    "full_report_morphology_term_count",
    "full_report_diffuse_ht_term_count",
    "first256_morphology_term_count",
    "first256_diffuse_ht_term_count",
    "latest_visit_morphology_term_count",
    "latest_visit_diffuse_ht_term_count",
    "latest_visit_negative_term_count",
    "full_report_negative_term_count",
    "first_positive_term_char_position",
    "first_diffuse_term_char_position",
    "evidence_exposed_in_first256",
    "positive_negative_overlap",
    "matched_morphology_terms",
    "matched_diffuse_terms",
    "matched_negative_terms",
    "matched_benign_terms",
    "non_thyroid_morphology_term_count",
]


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def report_text(row: Dict[str, Any]) -> str:
    for key in ("report_text", "text", "report", "reports_text", "raw_report_text"):
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def fmt(value: Any, digits: int = 4) -> str:
    try:
        if value is None or pd.isna(value):
            return "NA"
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "NA"


def frame_to_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    cols = [str(col) for col in frame.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in frame.iterrows():
        values: List[str] = []
        for col in frame.columns:
            value = row[col]
            if isinstance(value, float):
                values.append(fmt(value))
            else:
                try:
                    values.append("NA" if pd.isna(value) else str(value).replace("|", "/"))
                except (TypeError, ValueError):
                    values.append(str(value).replace("|", "/"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def term_matches(text: str, terms: Iterable[str]) -> List[str]:
    return [term for term in terms if term and term in text]


def any_term(text: str, terms: Iterable[str]) -> bool:
    return any(term and term in text for term in terms)


def thyroid_text(text: str) -> str:
    clauses = split_clauses(text)
    thyroid_clauses = [clause for clause in clauses if any_term(clause, THYROID_CUES)]
    return "。".join(thyroid_clauses)


def non_thyroid_text(text: str) -> str:
    clauses = split_clauses(text)
    non_thyroid_clauses = [clause for clause in clauses if any_term(clause, NON_THYROID_CUES)]
    return "。".join(non_thyroid_clauses)


def first_position(text: str, terms: Sequence[str]) -> int:
    positions = [text.find(term) for term in terms if term and text.find(term) >= 0]
    return min(positions) if positions else -1


def seed_from_path(path: Path) -> int:
    match = re.search(r"seed_(\d+)", path.name)
    return int(match.group(1)) if match else -1


def prob_column(frame: pd.DataFrame) -> str:
    for column in ("pred_prob", "prob", "prediction_prob", "score"):
        if column in frame.columns:
            return column
    raise ValueError("Prediction CSV must contain a probability column.")


def label_column(frame: pd.DataFrame) -> str | None:
    for column in ("pred_label", "prediction", "pred", "y_pred"):
        if column in frame.columns:
            return column
    return None


def prediction_files(run_dir: Path, split: str) -> List[Path]:
    direct = sorted((run_dir / "predictions").glob(f"{split}_predictions_seed_*.csv"))
    recursive = sorted(run_dir.glob(f"**/*{split}*prediction*seed_*.csv"))
    return sorted({path.resolve() for path in [*direct, *recursive] if path.is_file()})


def read_predictions(run_dir: Path, split: str, input_rows: List[Dict[str, str]]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    files = prediction_files(run_dir, split)
    if not files:
        input_rows.append({"path": str(run_dir), "status": f"missing_{split}_predictions", "notes": ""})
        return pd.DataFrame()
    for path in files:
        try:
            frame = pd.read_csv(path)
            if "patient_id" not in frame.columns:
                input_rows.append({"path": str(path), "status": "skipped", "notes": "missing patient_id"})
                continue
            frame = frame.copy()
            frame["patient_id"] = frame["patient_id"].astype(str)
            frame["split"] = frame["split"] if "split" in frame.columns else split
            frame["seed"] = frame["seed"] if "seed" in frame.columns else seed_from_path(path)
            frame["pred_prob"] = pd.to_numeric(frame[prob_column(frame)], errors="coerce")
            pred_col = label_column(frame)
            if pred_col:
                frame["pred_label"] = pd.to_numeric(frame[pred_col], errors="coerce")
            else:
                frame["pred_label"] = (frame["pred_prob"] >= 0.5).astype(int)
            frames.append(frame[["patient_id", "split", "seed", "pred_prob", "pred_label"]])
            input_rows.append({"path": str(path), "status": "loaded", "notes": f"{len(frame)} rows"})
        except Exception as exc:
            input_rows.append({"path": str(path), "status": "read_error", "notes": str(exc)})
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def manifest_frame(path: Path, input_rows: List[Dict[str, str]]) -> pd.DataFrame:
    rows = read_jsonl(path)
    input_rows.append({"path": str(path), "status": "loaded", "notes": f"{len(rows)} manifest rows"})
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["patient_id"] = frame["patient_id"].astype(str)
    frame["report_text"] = [report_text(row) for row in rows]
    for field in [
        "split",
        "label",
        "selected_n_visits",
        "used_images",
        "has_bio",
        "bio_missing_count",
        "matched_morphology_terms",
        "matched_negative_terms",
    ]:
        if field not in frame.columns:
            frame[field] = pd.NA
    keep = [
        "patient_id",
        "split",
        "label",
        "report_text",
        "selected_n_visits",
        "used_images",
        "has_bio",
        "bio_missing_count",
        "matched_morphology_terms",
        "matched_negative_terms",
    ]
    return frame[keep].drop_duplicates("patient_id")


def confusion_type(label: int, pred_label: int) -> str:
    if label == 1 and pred_label == 1:
        return "TP"
    if label == 1 and pred_label == 0:
        return "FN"
    if label == 0 and pred_label == 1:
        return "FP"
    return "TN"


def evidence_profile(text: str, text_max_length: int) -> Dict[str, Any]:
    model_text_window = max(int(text_max_length) - 2, 0)
    first_window = text[:model_text_window]
    full_thyroid = thyroid_text(text)
    first_thyroid = thyroid_text(first_window)
    visits = parse_visits(text)
    latest_block = visits[-1][1] if visits else text
    latest_thyroid = thyroid_text(latest_block)
    morphology_terms = term_matches(full_thyroid, MORPHOLOGY_CUES)
    diffuse_terms = term_matches(full_thyroid, DIFFUSE_HT_CUES)
    negative_terms = term_matches(full_thyroid, NEGATIVE_THYROID_CUES)
    benign_terms = term_matches(full_thyroid, BENIGN_NODULE_CUES)
    first_morph = count_terms(first_thyroid, MORPHOLOGY_CUES)
    first_diffuse = count_terms(first_thyroid, DIFFUSE_HT_CUES)
    first_negative = count_terms(first_thyroid, NEGATIVE_THYROID_CUES)
    return {
        "report_length_chars": len(text),
        "report_length_tokens_if_available": len(text.strip()) + 2 if text.strip() else 0,
        "audit_window_type": "model_char_token_window",
        "model_text_char_window": model_text_window,
        "full_report_morphology_term_count": count_terms(full_thyroid, MORPHOLOGY_CUES),
        "full_report_diffuse_ht_term_count": count_terms(full_thyroid, DIFFUSE_HT_CUES),
        "first256_morphology_term_count": first_morph,
        "first256_diffuse_ht_term_count": first_diffuse,
        "latest_visit_morphology_term_count": count_terms(latest_thyroid, MORPHOLOGY_CUES),
        "latest_visit_diffuse_ht_term_count": count_terms(latest_thyroid, DIFFUSE_HT_CUES),
        "latest_visit_negative_term_count": count_terms(latest_thyroid, NEGATIVE_THYROID_CUES),
        "full_report_negative_term_count": count_terms(full_thyroid, NEGATIVE_THYROID_CUES),
        "first_positive_term_char_position": first_position(text, list(MORPHOLOGY_CUES) + list(DIFFUSE_HT_CUES)),
        "first_diffuse_term_char_position": first_position(text, list(DIFFUSE_HT_CUES)),
        "evidence_exposed_in_first256": int((first_morph + first_diffuse) > 0),
        "positive_negative_overlap": int((count_terms(full_thyroid, MORPHOLOGY_CUES) + count_terms(full_thyroid, DIFFUSE_HT_CUES) > 0) and count_terms(full_thyroid, NEGATIVE_THYROID_CUES) > 0),
        "matched_morphology_terms": "|".join(morphology_terms),
        "matched_diffuse_terms": "|".join(diffuse_terms),
        "matched_negative_terms": "|".join(negative_terms),
        "matched_benign_terms": "|".join(benign_terms),
        "non_thyroid_morphology_term_count": count_terms(non_thyroid_text(text), MORPHOLOGY_CUES),
        "first_window_negative_term_count": first_negative,
    }


def build_positive_rows(preds: pd.DataFrame, manifest: pd.DataFrame, split: str, text_max_length: int) -> pd.DataFrame:
    merged = preds[preds["split"].astype(str).str.lower() == split].merge(manifest, on="patient_id", how="left", suffixes=("", "_manifest"))
    if "label_manifest" in merged.columns:
        merged["label"] = merged.get("label", pd.NA)
        merged["label"] = merged["label"].where(~merged["label"].isna(), merged["label_manifest"])
    merged["label"] = pd.to_numeric(merged["label"], errors="coerce")
    merged = merged[merged["label"] == 1].copy()
    if merged.empty:
        return merged
    merged["pred_label"] = pd.to_numeric(merged["pred_label"], errors="coerce").fillna((merged["pred_prob"] >= 0.5).astype(int)).astype(int)
    merged["confusion_type"] = [confusion_type(1, int(label)) for label in merged["pred_label"]]
    seed_summary = (
        merged.groupby("patient_id")
        .agg(
            cross_seed_fn_count=("confusion_type", lambda values: int((values == "FN").sum())),
            cross_seed_tp_count=("confusion_type", lambda values: int((values == "TP").sum())),
            cross_seed_pred_mean=("pred_prob", "mean"),
            cross_seed_pred_std=("pred_prob", "std"),
            n_pred_classes=("pred_label", "nunique"),
        )
        .reset_index()
    )
    seed_summary["cross_seed_pred_std"] = seed_summary["cross_seed_pred_std"].fillna(0.0)
    seed_summary["seed_sensitive"] = ((seed_summary["n_pred_classes"] > 1) | (seed_summary["cross_seed_pred_std"] >= 0.15)).astype(int)
    seed_summary["case_type"] = "seed_sensitive"
    seed_summary.loc[seed_summary["cross_seed_fn_count"] >= 2, "case_type"] = "stable_fn"
    seed_summary.loc[seed_summary["cross_seed_tp_count"] >= 2, "case_type"] = "stable_tp"
    merged = merged.merge(seed_summary, on="patient_id", how="left")
    profiles = [evidence_profile(str(text), text_max_length=text_max_length) for text in merged["report_text"].fillna("")]
    prof = pd.DataFrame(profiles)
    for duplicate_col in ("matched_morphology_terms", "matched_negative_terms"):
        if duplicate_col in merged.columns and duplicate_col in prof.columns:
            merged = merged.drop(columns=[duplicate_col])
    out = pd.concat([merged.reset_index(drop=True), prof.reset_index(drop=True)], axis=1)
    for col in PATIENT_OUTPUT_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    return out[PATIENT_OUTPUT_COLUMNS].sort_values(["case_type", "patient_id", "seed"])


def summarize_groups(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    fields = [
        "first256_morphology_term_count",
        "first256_diffuse_ht_term_count",
        "full_report_morphology_term_count",
        "full_report_diffuse_ht_term_count",
        "latest_visit_morphology_term_count",
        "latest_visit_diffuse_ht_term_count",
        "full_report_negative_term_count",
        "positive_negative_overlap",
        "report_length_chars",
        "selected_n_visits",
        "cross_seed_pred_std",
    ]
    tmp = rows.copy()
    tmp["row_type"] = tmp["confusion_type"]
    stable = tmp.drop_duplicates("patient_id").copy()
    stable["row_type"] = stable["case_type"]
    both = pd.concat([tmp, stable], ignore_index=True)
    grouped = both.groupby("row_type", dropna=False)
    out = grouped.agg(n_rows=("patient_id", "count"), n_patients=("patient_id", "nunique"), mean_pred_prob=("pred_prob", "mean")).reset_index()
    for field in fields:
        if field in both.columns:
            vals = grouped[field].mean().reset_index(name=f"mean_{field}")
            out = out.merge(vals, on="row_type", how="left")
    return out.sort_values(["row_type"])


def stable_fn_cases(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    stable = rows[rows["case_type"] == "stable_fn"].copy()
    keep = [
        "patient_id",
        "cross_seed_fn_count",
        "cross_seed_tp_count",
        "cross_seed_pred_mean",
        "cross_seed_pred_std",
        "first256_morphology_term_count",
        "first256_diffuse_ht_term_count",
        "full_report_diffuse_ht_term_count",
        "latest_visit_diffuse_ht_term_count",
        "full_report_negative_term_count",
        "positive_negative_overlap",
        "report_length_chars",
        "selected_n_visits",
        "matched_morphology_terms",
        "matched_diffuse_terms",
        "matched_negative_terms",
    ]
    return stable.drop_duplicates("patient_id")[keep].sort_values(["first256_diffuse_ht_term_count", "cross_seed_pred_mean", "report_length_chars"])


def seed_sensitive_cases(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    sens = rows[rows["seed_sensitive"] == 1].copy()
    keep = [
        "patient_id",
        "case_type",
        "cross_seed_fn_count",
        "cross_seed_tp_count",
        "cross_seed_pred_mean",
        "cross_seed_pred_std",
        "first256_diffuse_ht_term_count",
        "full_report_diffuse_ht_term_count",
        "report_length_chars",
        "selected_n_visits",
    ]
    return sens.drop_duplicates("patient_id")[keep].sort_values(["cross_seed_pred_std", "cross_seed_fn_count"], ascending=[False, False])


def exposure_strata(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    strata_defs = {
        "diffuse_exposed_first_window": rows["first256_diffuse_ht_term_count"] > 0,
        "only_generic_morphology_exposed": (rows["first256_morphology_term_count"] > 0) & (rows["first256_diffuse_ht_term_count"] == 0),
        "no_positive_thyroid_evidence_exposed": (rows["first256_morphology_term_count"] == 0) & (rows["first256_diffuse_ht_term_count"] == 0),
        "positive_negative_overlap_full_report": rows["positive_negative_overlap"] == 1,
    }
    out_rows: List[Dict[str, Any]] = []
    for name, mask in strata_defs.items():
        group = rows[mask].copy()
        out_rows.append(
            {
                "stratum": name,
                "n_rows": len(group),
                "n_patients": group["patient_id"].nunique(),
                "fn_count": int((group["confusion_type"] == "FN").sum()) if not group.empty else 0,
                "fn_rate": float((group["confusion_type"] == "FN").mean()) if not group.empty else 0.0,
                "mean_pred_prob": group["pred_prob"].mean() if not group.empty else 0.0,
                "mean_cross_seed_pred_std": group["cross_seed_pred_std"].mean() if not group.empty else 0.0,
            }
        )
    return pd.DataFrame(out_rows)


def seed_overlap(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    fn_by_seed = {
        int(seed): set(group[group["confusion_type"] == "FN"]["patient_id"].astype(str))
        for seed, group in rows.groupby("seed")
    }
    out_rows: List[Dict[str, Any]] = []
    for seed, ids in sorted(fn_by_seed.items()):
        others = set().union(*(other for other_seed, other in fn_by_seed.items() if other_seed != seed))
        out_rows.append({"comparison": f"seed_{seed}", "fn_count": len(ids), "unique_fn_count": len(ids - others), "overlap_count": len(ids & others)})
    for left, right in itertools.combinations(sorted(fn_by_seed), 2):
        out_rows.append(
            {
                "comparison": f"seed_{left}_vs_seed_{right}",
                "fn_count": len(fn_by_seed[left]),
                "unique_fn_count": len(fn_by_seed[left] ^ fn_by_seed[right]),
                "overlap_count": len(fn_by_seed[left] & fn_by_seed[right]),
            }
        )
    all_sets = list(fn_by_seed.values())
    if all_sets:
        out_rows.append({"comparison": "all_seed_intersection", "fn_count": len(set.intersection(*all_sets)), "unique_fn_count": 0, "overlap_count": len(set.intersection(*all_sets))})
    return pd.DataFrame(out_rows)


def decide(rows: pd.DataFrame) -> tuple[str, Dict[str, Any]]:
    stable = rows.drop_duplicates("patient_id").copy()
    stable_fn = stable[stable["case_type"] == "stable_fn"]
    stable_tp = stable[stable["case_type"] == "stable_tp"]
    metrics = {
        "stable_fn_patients": int(len(stable_fn)),
        "stable_tp_patients": int(len(stable_tp)),
        "stable_fn_mean_first256_diffuse": float(stable_fn["first256_diffuse_ht_term_count"].mean()) if len(stable_fn) else 0.0,
        "stable_tp_mean_first256_diffuse": float(stable_tp["first256_diffuse_ht_term_count"].mean()) if len(stable_tp) else 0.0,
        "stable_fn_no_diffuse_rate": float((stable_fn["first256_diffuse_ht_term_count"] == 0).mean()) if len(stable_fn) else 0.0,
        "stable_fn_exposed_positive_rate": float((stable_fn["evidence_exposed_in_first256"] == 1).mean()) if len(stable_fn) else 0.0,
        "stable_fn_mean_full_diffuse": float(stable_fn["full_report_diffuse_ht_term_count"].mean()) if len(stable_fn) else 0.0,
    }
    if not len(stable_fn):
        return "MIXED_OR_INCONCLUSIVE", metrics
    not_exposed = metrics["stable_fn_no_diffuse_rate"] >= 0.60 and metrics["stable_fn_mean_full_diffuse"] > metrics["stable_fn_mean_first256_diffuse"]
    exposed_but_low = metrics["stable_fn_exposed_positive_rate"] >= 0.70 and metrics["stable_fn_mean_first256_diffuse"] >= 1.0
    if not_exposed and not exposed_but_low:
        return "EVIDENCE_NOT_EXPOSED", metrics
    if exposed_but_low and not not_exposed:
        return "EVIDENCE_EXPOSED_BUT_NOT_USED", metrics
    return "MIXED_OR_INCONCLUSIVE", metrics


def write_report(out_dir: Path, val_rows: pd.DataFrame, test_rows: pd.DataFrame, decision: str, decision_metrics: Dict[str, Any]) -> None:
    summary = summarize_groups(val_rows)
    strata = exposure_strata(val_rows)
    overlap = seed_overlap(val_rows)
    stable_cases = stable_fn_cases(val_rows).head(20)
    lines = [
        "# Phase C14-A FN Token Exposure Audit",
        "",
        "This is an analysis-only audit. No training, threshold tuning, label editing, split editing, or model-code changes were performed.",
        "",
        "## Audit Window",
        "",
        "- The project tokenizer is character-level with special tokens.",
        "- `text_max_length=256` means approximately 254 report characters are visible as text tokens.",
        "- Fields named `first256_*` use this model character-token window, not a word tokenizer.",
        "",
        "## Validation Positive Cohorts",
        "",
        frame_to_markdown(summary),
        "",
        "## Evidence Exposure Strata",
        "",
        frame_to_markdown(strata),
        "",
        "## Seed FN Overlap",
        "",
        frame_to_markdown(overlap),
        "",
        "## Stable FN Examples",
        "",
        frame_to_markdown(stable_cases),
        "",
        "## Decision Metrics",
        "",
        frame_to_markdown(pd.DataFrame([decision_metrics])),
        "",
        "## Decision",
        "",
        f"`{decision}`.",
    ]
    if decision == "EVIDENCE_NOT_EXPOSED":
        lines.extend(
            [
                "",
                "Next route: C14-B balanced temporal evidence prefix pilot is allowed as a design proposal only after a positive-preservation audit plan is specified.",
            ]
        )
    elif decision == "EVIDENCE_EXPOSED_BUT_NOT_USED":
        lines.extend(
            [
                "",
                "Next route: stop report-order changes and run analysis-first text representation / fusion contribution audits.",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "Next route: no training. Run finer manual case review or attribution audit before changing report construction or model internals.",
            ]
        )
    lines.extend(["", f"Test reporting-only positive rows generated: {len(test_rows)}. They were not used for the decision."])
    (out_dir / "c14a_token_exposure_audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "phase_c14a_final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase C14-A morphology-positive FN token exposure audit.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--text-max-length", type=int, default=256)
    parser.add_argument("--include-test-reporting-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    input_rows: List[Dict[str, str]] = []
    manifest = manifest_frame(Path(args.manifest), input_rows)
    run_dir = Path(args.run_dir)
    val_preds = read_predictions(run_dir, "val", input_rows)
    test_preds = read_predictions(run_dir, "test", input_rows) if args.include_test_reporting_only else pd.DataFrame()
    val_rows = build_positive_rows(val_preds, manifest, "val", text_max_length=int(args.text_max_length))
    test_rows = build_positive_rows(test_preds, manifest, "test", text_max_length=int(args.text_max_length)) if not test_preds.empty else pd.DataFrame()
    decision, decision_metrics = decide(val_rows)

    val_rows.to_csv(out_dir / "c14a_positive_patient_token_exposure_val.csv", index=False)
    summarize_groups(val_rows).to_csv(out_dir / "c14a_fn_vs_tp_summary_val.csv", index=False)
    stable_fn_cases(val_rows).to_csv(out_dir / "c14a_cross_seed_stable_fn_cases_val.csv", index=False)
    seed_sensitive_cases(val_rows).to_csv(out_dir / "c14a_seed_sensitive_positive_cases_val.csv", index=False)
    exposure_strata(val_rows).to_csv(out_dir / "c14a_evidence_exposure_strata_val.csv", index=False)
    seed_overlap(val_rows).to_csv(out_dir / "c14a_seed_overlap_summary_val.csv", index=False)
    if args.include_test_reporting_only:
        test_rows.to_csv(out_dir / "c14a_positive_patient_token_exposure_test_reporting_only.csv", index=False)
    pd.DataFrame(input_rows).to_csv(out_dir / "inputs_used_and_missing.csv", index=False)
    write_report(out_dir, val_rows=val_rows, test_rows=test_rows, decision=decision, decision_metrics=decision_metrics)
    print(json.dumps({"output_dir": str(out_dir), "decision": decision, **decision_metrics}, ensure_ascii=False))


if __name__ == "__main__":
    main()
