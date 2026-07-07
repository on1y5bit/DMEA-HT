from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dmea_ht.data import read_manifest


THYROID_CUES = ["甲状腺", "峡部", "左侧叶", "右侧叶", "双侧叶", "两侧叶", "左叶", "右叶"]
NON_THYROID_CUES = [
    "乳房",
    "乳腺",
    "颈动脉",
    "心脏",
    "二尖瓣",
    "主动脉",
    "三尖瓣",
    "肺动脉",
    "肝",
    "胆囊",
    "脾",
    "胰腺",
    "肾",
    "膀胱",
    "子宫",
    "附件",
    "前列腺",
]
MORPHOLOGY_CUES = [
    "弥漫性",
    "实质回声不均",
    "回声不均",
    "回声欠均",
    "回声欠均匀",
    "回声粗糙",
    "低回声",
    "体积增大",
    "表面欠光滑",
    "桥本",
    "甲状腺炎",
]
DIFFUSE_HT_CUES = ["弥漫性", "实质回声不均", "回声不均", "回声粗糙", "体积增大", "表面欠光滑", "桥本", "甲状腺炎"]
BENIGN_NODULE_CUES = [
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
]
NEGATIVE_THYROID_CUES = [
    "回声均匀",
    "实质回声均匀",
    "内部回声细小均匀",
    "未见明显异常回声",
    "未见异常回声",
    "大小正常",
    "形态大小正常",
    "未见异常血流",
]

HYPOTHESES = [
    "latest_negative_suppresses_history",
    "benign_nodule_without_latest_diffuse",
    "require_latest_diffuse_ht_like",
    "non_thyroid_morphology_only",
    "positive_negative_overlap_review",
]


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
            elif isinstance(value, (list, tuple, set)):
                values.append(str(list(value)).replace("|", "/"))
            elif isinstance(value, dict):
                values.append(json.dumps(value, ensure_ascii=False).replace("|", "/"))
            else:
                try:
                    values.append("NA" if pd.isna(value) else str(value).replace("|", "/"))
                except (TypeError, ValueError):
                    values.append(str(value).replace("|", "/"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def any_term(text: str, terms: Iterable[str]) -> bool:
    return any(term and term in text for term in terms)


def count_terms(text: str, terms: Iterable[str]) -> int:
    return sum(1 for term in terms if term and term in text)


def parse_visits(text: str) -> List[Tuple[str, str]]:
    if not text:
        return []
    matches = list(re.finditer(r"\[(\d{4}-\d{2}-\d{2})\]", text))
    if not matches:
        return [("unknown", text.strip())] if text.strip() else []
    visits: List[Tuple[str, str]] = []
    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        visits.append((match.group(1), text[start:end].strip()))
    return visits


def split_clauses(text: str) -> List[str]:
    pieces = re.split(r"[。；;\n]+", text or "")
    return [piece.strip() for piece in pieces if piece.strip()]


def thyroid_text(block: str) -> str:
    clauses = split_clauses(block)
    return "。".join(clause for clause in clauses if any_term(clause, THYROID_CUES))


def non_thyroid_text(block: str) -> str:
    clauses = split_clauses(block)
    return "。".join(clause for clause in clauses if any_term(clause, NON_THYROID_CUES))


def read_manifest_frame(path: Path) -> pd.DataFrame:
    frame = pd.DataFrame(read_manifest(path))
    frame["patient_id"] = frame["patient_id"].astype(str)
    for field in ["patient_id", "split", "label", "report_text"]:
        if field not in frame.columns:
            frame[field] = pd.NA
    return frame.drop_duplicates("patient_id")


def seed_from_path(path: Path) -> int:
    match = re.search(r"seed_(\d+)", path.name)
    return int(match.group(1)) if match else -1


def prob_column(frame: pd.DataFrame) -> str:
    for column in ("pred_prob", "prob", "prediction_prob", "score"):
        if column in frame.columns:
            return column
    raise ValueError("Prediction CSV must contain pred_prob/prob.")


def read_predictions(run_dir: Path, split: str) -> pd.DataFrame:
    paths = sorted((run_dir / "predictions").glob(f"{split}_predictions_seed_*.csv"))
    frames: List[pd.DataFrame] = []
    for path in paths:
        frame = pd.read_csv(path)
        if "patient_id" not in frame.columns:
            continue
        frame["patient_id"] = frame["patient_id"].astype(str)
        frame["seed"] = frame["seed"] if "seed" in frame.columns else seed_from_path(path)
        frame["split"] = frame["split"] if "split" in frame.columns else split
        frame["pred_prob"] = pd.to_numeric(frame[prob_column(frame)], errors="coerce")
        frames.append(frame[["patient_id", "seed", "split", "label", "pred_prob"]])
    if not frames:
        raise FileNotFoundError(f"No {split} prediction CSVs found under {run_dir / 'predictions'}")
    return pd.concat(frames, ignore_index=True)


def patient_prediction_summary(preds: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    pred = preds.copy()
    pred["pred_label_seed"] = (pred["pred_prob"] >= 0.5).astype(int)
    summary = (
        pred.groupby("patient_id", as_index=False)
        .agg(
            seed_count=("seed", "nunique"),
            mean_pred_prob=("pred_prob", "mean"),
            max_pred_prob=("pred_prob", "max"),
            min_pred_prob=("pred_prob", "min"),
            n_seed_pred_positive=("pred_label_seed", "sum"),
        )
    )
    merged = summary.merge(manifest, on="patient_id", how="left", suffixes=("", "_manifest"))
    if "label_manifest" in merged.columns:
        merged["label"] = merged["label_manifest"]
    merged["label"] = pd.to_numeric(merged["label"], errors="coerce").astype(int)
    merged["mean_pred_label"] = (merged["mean_pred_prob"] >= 0.5).astype(int)
    merged["any_seed_pred_positive"] = (merged["n_seed_pred_positive"] > 0).astype(int)
    merged["mean_confusion_type"] = merged.apply(mean_confusion_type, axis=1)
    return merged


def mean_confusion_type(row: pd.Series) -> str:
    label = int(row["label"])
    pred = int(row["mean_pred_label"])
    if label == 1 and pred == 1:
        return "TP"
    if label == 0 and pred == 0:
        return "TN"
    if label == 0 and pred == 1:
        return "FP"
    return "FN"


def report_text(row: pd.Series) -> str:
    for field in ("report_text", "text", "report", "reports_text", "raw_report_text"):
        if field in row and not pd.isna(row[field]):
            return str(row[field])
    return ""


def evidence_profile(row: pd.Series) -> Dict[str, Any]:
    text = report_text(row)
    visits = parse_visits(text)
    if not visits:
        visits = [("unknown", text)]
    visit_profiles: List[Dict[str, Any]] = []
    for idx, (date, block) in enumerate(visits):
        thyroid = thyroid_text(block)
        non_thyroid = non_thyroid_text(block)
        visit_profiles.append(
            {
                "idx": idx,
                "date": date,
                "thyroid_morphology": count_terms(thyroid, MORPHOLOGY_CUES),
                "thyroid_diffuse": count_terms(thyroid, DIFFUSE_HT_CUES),
                "thyroid_benign": count_terms(thyroid, BENIGN_NODULE_CUES),
                "thyroid_negative": count_terms(thyroid, NEGATIVE_THYROID_CUES),
                "non_thyroid_morphology": count_terms(non_thyroid, MORPHOLOGY_CUES),
                "thyroid_preview": thyroid[:500],
            }
        )
    latest = visit_profiles[-1]
    earlier = visit_profiles[:-1]
    any_earlier_morph = any(item["thyroid_morphology"] > 0 for item in earlier)
    any_thyroid_morph = any(item["thyroid_morphology"] > 0 for item in visit_profiles)
    any_non_thyroid_morph = any(item["non_thyroid_morphology"] > 0 for item in visit_profiles)
    any_overlap = any(item["thyroid_morphology"] > 0 and item["thyroid_negative"] > 0 for item in visit_profiles)
    any_benign = any(item["thyroid_benign"] > 0 for item in visit_profiles)
    any_diffuse = any(item["thyroid_diffuse"] > 0 for item in visit_profiles)
    flags = {
        "n_visits_parsed": len(visit_profiles),
        "any_thyroid_morphology": int(any_thyroid_morph),
        "any_non_thyroid_morphology": int(any_non_thyroid_morph),
        "any_thyroid_positive_negative_overlap": int(any_overlap),
        "any_benign_nodule_signal": int(any_benign),
        "any_diffuse_ht_like_signal": int(any_diffuse),
        "latest_thyroid_morphology": latest["thyroid_morphology"],
        "latest_thyroid_diffuse": latest["thyroid_diffuse"],
        "latest_thyroid_benign": latest["thyroid_benign"],
        "latest_thyroid_negative": latest["thyroid_negative"],
        "earlier_thyroid_morphology_present": int(any_earlier_morph),
        "latest_visit_date": latest["date"],
        "latest_thyroid_preview": latest["thyroid_preview"],
    }
    flags["latest_negative_suppresses_history"] = int(any_earlier_morph and latest["thyroid_negative"] > 0)
    flags["benign_nodule_without_latest_diffuse"] = int(any_benign and latest["thyroid_diffuse"] == 0)
    flags["require_latest_diffuse_ht_like"] = int(any_thyroid_morph and latest["thyroid_diffuse"] == 0)
    flags["non_thyroid_morphology_only"] = int(any_non_thyroid_morph and not any_thyroid_morph)
    flags["positive_negative_overlap_review"] = int(any_overlap)
    return flags


def build_patient_table(preds: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    patients = patient_prediction_summary(preds, manifest)
    rows: List[Dict[str, Any]] = []
    for _, row in patients.iterrows():
        profile = evidence_profile(row)
        out = {
            "patient_id": row["patient_id"],
            "split": row.get("split", "val"),
            "label": int(row["label"]),
            "seed_count": int(row["seed_count"]),
            "mean_pred_prob": row["mean_pred_prob"],
            "max_pred_prob": row["max_pred_prob"],
            "n_seed_pred_positive": int(row["n_seed_pred_positive"]),
            "mean_confusion_type": row["mean_confusion_type"],
        }
        out.update(profile)
        rows.append(out)
    return pd.DataFrame(rows)


def summarize_hypotheses(patient_table: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if patient_table.empty:
        return pd.DataFrame()
    fp = patient_table[patient_table["mean_confusion_type"] == "FP"]
    label1 = patient_table[patient_table["label"] == 1]
    tp = patient_table[patient_table["mean_confusion_type"] == "TP"]
    fn = patient_table[patient_table["mean_confusion_type"] == "FN"]
    for hyp in HYPOTHESES:
        flagged = patient_table[patient_table[hyp] > 0]
        flagged_fp = fp[fp[hyp] > 0]
        flagged_label1 = label1[label1[hyp] > 0]
        flagged_tp = tp[tp[hyp] > 0]
        flagged_fn = fn[fn[hyp] > 0]
        rows.append(
            {
                "hypothesis": hyp,
                "n_flagged_all_val": len(flagged),
                "n_flagged_mean_fp": len(flagged_fp),
                "mean_fp_capture_rate": len(flagged_fp) / max(len(fp), 1),
                "n_flagged_label1_positive": len(flagged_label1),
                "positive_patient_flag_rate": len(flagged_label1) / max(len(label1), 1),
                "n_flagged_true_positive": len(flagged_tp),
                "true_positive_flag_rate": len(flagged_tp) / max(len(tp), 1),
                "n_flagged_false_negative": len(flagged_fn),
                "precision_for_mean_fp_among_flagged": len(flagged_fp) / max(len(flagged), 1),
                "recommendation_status": hypothesis_status(len(flagged_fp), len(fp), len(flagged_label1), len(label1), hyp),
            }
        )
    return pd.DataFrame(rows).sort_values(["recommendation_status", "mean_fp_capture_rate"], ascending=[True, False])


def hypothesis_status(flagged_fp: int, total_fp: int, flagged_pos: int, total_pos: int, hyp: str) -> str:
    fp_rate = flagged_fp / max(total_fp, 1)
    pos_rate = flagged_pos / max(total_pos, 1)
    if fp_rate >= 0.35 and pos_rate <= 0.20:
        return "PILOT_ELIGIBLE_LOW_POSITIVE_RISK"
    if fp_rate >= 0.35 and pos_rate <= 0.40:
        return "AUDIT_MORE_MEDIUM_POSITIVE_RISK"
    if fp_rate >= 0.20:
        return "AUDIT_MORE_HIGH_POSITIVE_RISK"
    if hyp == "non_thyroid_morphology_only" and flagged_fp > 0:
        return "CASE_REVIEW_ONLY"
    return "NOT_ENOUGH_FP_COVERAGE"


def positive_risk_table(patient_table: pd.DataFrame) -> pd.DataFrame:
    label1 = patient_table[patient_table["label"] == 1]
    rows: List[Dict[str, Any]] = []
    for hyp in HYPOTHESES:
        flagged = label1[label1[hyp] > 0].copy()
        rows.append(
            {
                "hypothesis": hyp,
                "n_positive_flagged": len(flagged),
                "mean_pred_prob_positive_flagged": flagged["mean_pred_prob"].mean() if not flagged.empty else pd.NA,
                "n_true_positive_flagged": int((flagged["mean_confusion_type"] == "TP").sum()) if not flagged.empty else 0,
                "n_false_negative_flagged": int((flagged["mean_confusion_type"] == "FN").sum()) if not flagged.empty else 0,
                "example_positive_patient_ids": ",".join(flagged["patient_id"].head(10).astype(str).tolist()),
            }
        )
    return pd.DataFrame(rows)


def write_report(patient_table: pd.DataFrame, summary: pd.DataFrame, risk: pd.DataFrame, out_dir: Path) -> str:
    fp = patient_table[patient_table["mean_confusion_type"] == "FP"]
    label1 = patient_table[patient_table["label"] == 1]
    eligible = summary[summary["recommendation_status"] == "PILOT_ELIGIBLE_LOW_POSITIVE_RISK"]
    if not eligible.empty:
        recommendation = "ALLOW_REPORT_FILTER_PILOT_FOR_LOW_RISK_HYPOTHESIS"
    elif (summary["recommendation_status"] == "AUDIT_MORE_MEDIUM_POSITIVE_RISK").any():
        recommendation = "CONTINUE_AUDIT_BEFORE_REPORT_FILTER_PILOT"
    else:
        recommendation = "NO_REPORT_FILTER_PILOT_YET"
    top_cases = fp.sort_values("mean_pred_prob", ascending=False)[
        [
            "patient_id",
            "mean_pred_prob",
            "latest_negative_suppresses_history",
            "benign_nodule_without_latest_diffuse",
            "require_latest_diffuse_ht_like",
            "non_thyroid_morphology_only",
            "positive_negative_overlap_review",
            "latest_visit_date",
            "latest_thyroid_preview",
        ]
    ].head(15)
    lines = [
        "# Phase C11 Report-Filter Hypothesis Audit",
        "",
        "Phase C11 is analysis-only. It audits report-construction hypotheses before any training pilot.",
        "",
        "## Validation Cohort",
        "",
        f"- Validation patients audited: {len(patient_table)}.",
        f"- Mean-threshold false-positive patients: {len(fp)}.",
        f"- Label-positive patients for positive-preservation risk: {len(label1)}.",
        "",
        "## Hypothesis Summary",
        "",
        frame_to_markdown(summary),
        "",
        "## Positive-Preservation Risk",
        "",
        frame_to_markdown(risk),
        "",
        "## Highest-Probability FP Examples",
        "",
        frame_to_markdown(top_cases),
        "",
        "## Interpretation",
        "",
        "- A hypothesis is not a model change; it is only eligible for a later low-cost pilot if it captures FP cases without flagging many label-positive patients.",
        "- Test data is not used here.",
        "- Shortcut fields remain audit-only and are not candidate model inputs.",
        "",
        "## Recommendation",
        "",
        f"`{recommendation}`.",
        "",
        "If a later pilot is allowed, it must be report-construction only, validation-selected, bad-seed/stress-seed checked, and followed by positive-preservation and shortcut residual audits.",
    ]
    (out_dir / "phase_c11_final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return recommendation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit report-filter hypotheses before any C11/C12 pilot.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="val")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = read_manifest_frame(Path(args.manifest))
    manifest = manifest[manifest["split"].astype(str) == args.split].copy()
    preds = read_predictions(Path(args.run_dir), args.split)
    patient_table = build_patient_table(preds, manifest)
    summary = summarize_hypotheses(patient_table)
    risk = positive_risk_table(patient_table)
    patient_table.to_csv(out_dir / "c11_report_filter_patient_table_val.csv", index=False)
    summary.to_csv(out_dir / "c11_report_filter_hypothesis_summary_val.csv", index=False)
    risk.to_csv(out_dir / "c11_positive_preservation_risk_val.csv", index=False)
    recommendation = write_report(patient_table, summary, risk, out_dir)
    print(f"Wrote Phase C11 report-filter hypothesis audit to {out_dir}")
    print(f"Recommendation: {recommendation}")


if __name__ == "__main__":
    main()
