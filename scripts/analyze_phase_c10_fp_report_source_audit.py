from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dmea_ht.data import parse_maybe_list, read_manifest


THYROID_CUES = [
    "甲状腺",
    "峡部",
    "左侧叶",
    "右侧叶",
    "双侧叶",
    "两侧叶",
    "左叶",
    "右叶",
]
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
DIFFUSE_HT_CUES = [
    "弥漫性",
    "实质回声不均",
    "回声不均",
    "回声欠均",
    "回声粗糙",
    "体积增大",
    "表面欠光滑",
    "桥本",
    "甲状腺炎",
]
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
                    if pd.isna(value):
                        values.append("NA")
                    else:
                        values.append(str(value).replace("|", "/"))
                except (TypeError, ValueError):
                    values.append(str(value).replace("|", "/"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item).strip()]
    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        pass
    parsed = parse_maybe_list(value)
    if isinstance(parsed, list):
        return [str(item) for item in parsed if str(item).strip()]
    text = str(value).strip()
    if not text or text in {"[]", "nan", "None", "NA"}:
        return []
    return [text]


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
        block = text[start:end].strip()
        visits.append((match.group(1), block))
    return visits


def split_clauses(text: str) -> List[str]:
    if not text:
        return []
    pieces = re.split(r"[。；;\n]+", text)
    return [piece.strip() for piece in pieces if piece.strip()]


def thyroid_clauses(clauses: List[str]) -> List[str]:
    return [clause for clause in clauses if any_term(clause, THYROID_CUES)]


def non_thyroid_clauses(clauses: List[str]) -> List[str]:
    return [clause for clause in clauses if any_term(clause, NON_THYROID_CUES)]


def read_manifest_frame(path: Path) -> pd.DataFrame:
    frame = pd.DataFrame(read_manifest(path))
    frame["patient_id"] = frame["patient_id"].astype(str)
    for field in ["patient_id", "label", "split", "report_text"]:
        if field not in frame.columns:
            frame[field] = pd.NA
    return frame.drop_duplicates("patient_id")


def read_patient_audit(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["patient_id"] = frame["patient_id"].astype(str)
    return frame


def report_text(row: pd.Series) -> str:
    for field in ("report_text", "text", "report", "reports_text", "raw_report_text"):
        if field in row and not pd.isna(row[field]):
            return str(row[field])
    return str(row.get("report_text_preview", ""))


def visit_rows(patient_audit: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    merged = patient_audit.merge(manifest, on="patient_id", how="left", suffixes=("", "_manifest"))
    rows: List[Dict[str, Any]] = []
    for _, row in merged.iterrows():
        text = report_text(row)
        morphology_terms = as_list(row.get("matched_morphology_terms"))
        negative_terms = as_list(row.get("matched_negative_terms"))
        visits = parse_visits(text)
        for visit_idx, (visit_date, block) in enumerate(visits):
            clauses = split_clauses(block)
            thyroid = thyroid_clauses(clauses)
            non_thyroid = non_thyroid_clauses(clauses)
            thyroid_text = "。".join(thyroid)
            non_thyroid_text = "。".join(non_thyroid)
            morph_in_thyroid = count_terms(thyroid_text, morphology_terms)
            morph_in_non_thyroid = count_terms(non_thyroid_text, morphology_terms)
            negative_in_thyroid = count_terms(thyroid_text, negative_terms)
            ht_like = count_terms(thyroid_text, DIFFUSE_HT_CUES)
            benign_like = count_terms(thyroid_text, BENIGN_NODULE_CUES)
            thyroid_negative = count_terms(thyroid_text, NEGATIVE_THYROID_CUES)
            rows.append(
                {
                    "patient_id": row["patient_id"],
                    "visit_index": visit_idx,
                    "visit_date": visit_date,
                    "label": row.get("label"),
                    "mean_pred_prob": row.get("mean_pred_prob"),
                    "max_pred_prob": row.get("max_pred_prob"),
                    "n_unique_fp_seeds": row.get("n_unique_fp_seeds"),
                    "high_confidence_fp_seed_count": row.get("high_confidence_fp_seed_count"),
                    "selected_n_visits": row.get("selected_n_visits"),
                    "report_length": row.get("report_length"),
                    "morphology_terms": morphology_terms,
                    "negative_terms": negative_terms,
                    "n_clauses": len(clauses),
                    "n_thyroid_clauses": len(thyroid),
                    "n_non_thyroid_clauses": len(non_thyroid),
                    "morphology_term_hits_in_thyroid_clauses": morph_in_thyroid,
                    "morphology_term_hits_in_non_thyroid_clauses": morph_in_non_thyroid,
                    "negative_term_hits_in_thyroid_clauses": negative_in_thyroid,
                    "ht_like_thyroid_cue_count": ht_like,
                    "benign_nodule_thyroid_cue_count": benign_like,
                    "negative_thyroid_cue_count": thyroid_negative,
                    "thyroid_positive_and_negative_overlap": int(morph_in_thyroid > 0 and (negative_in_thyroid > 0 or thyroid_negative > 0)),
                    "non_thyroid_morphology_source_suspected": int(morph_in_non_thyroid > 0 and morph_in_thyroid == 0),
                    "benign_nodule_mimic_suspected": int(benign_like > 0 and ht_like == 0),
                    "diffuse_ht_like_signal_present": int(ht_like > 0),
                    "thyroid_text_preview": thyroid_text[:600],
                    "non_thyroid_text_preview": non_thyroid_text[:400],
                }
            )
    return pd.DataFrame(rows)


def patient_summary(visit_frame: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if visit_frame.empty:
        return pd.DataFrame()
    for patient_id, group in visit_frame.groupby("patient_id"):
        first = group.iloc[0]
        latest = group.sort_values("visit_index").iloc[-1]
        early = group.sort_values("visit_index").iloc[0]
        rows.append(
            {
                "patient_id": patient_id,
                "n_visits_parsed": len(group),
                "mean_pred_prob": first.get("mean_pred_prob"),
                "max_pred_prob": first.get("max_pred_prob"),
                "n_unique_fp_seeds": first.get("n_unique_fp_seeds"),
                "high_confidence_fp_seed_count": first.get("high_confidence_fp_seed_count"),
                "any_thyroid_morphology_hit": int((group["morphology_term_hits_in_thyroid_clauses"] > 0).any()),
                "any_non_thyroid_morphology_hit": int((group["morphology_term_hits_in_non_thyroid_clauses"] > 0).any()),
                "any_thyroid_positive_negative_overlap": int((group["thyroid_positive_and_negative_overlap"] > 0).any()),
                "any_benign_nodule_mimic": int((group["benign_nodule_mimic_suspected"] > 0).any()),
                "any_diffuse_ht_like_signal": int((group["diffuse_ht_like_signal_present"] > 0).any()),
                "latest_visit_date": latest.get("visit_date"),
                "latest_visit_has_thyroid_morphology_hit": int(latest.get("morphology_term_hits_in_thyroid_clauses", 0) > 0),
                "latest_visit_has_negative_thyroid_cue": int(latest.get("negative_term_hits_in_thyroid_clauses", 0) > 0 or latest.get("negative_thyroid_cue_count", 0) > 0),
                "early_positive_latest_negative_conflict": int(
                    early.get("morphology_term_hits_in_thyroid_clauses", 0) > 0
                    and (latest.get("negative_term_hits_in_thyroid_clauses", 0) > 0 or latest.get("negative_thyroid_cue_count", 0) > 0)
                ),
                "source_audit_priority": source_priority(group),
                "recommended_manual_review_focus": review_focus(group),
            }
        )
    return pd.DataFrame(rows).sort_values(["source_audit_priority", "max_pred_prob"], ascending=[False, False])


def source_priority(group: pd.DataFrame) -> int:
    score = 0
    if int(group.iloc[0].get("high_confidence_fp_seed_count", 0)) > 0:
        score += 3
    if (group["thyroid_positive_and_negative_overlap"] > 0).any():
        score += 3
    if (group["benign_nodule_mimic_suspected"] > 0).any():
        score += 2
    if (group["non_thyroid_morphology_source_suspected"] > 0).any():
        score += 2
    latest = group.sort_values("visit_index").iloc[-1]
    early = group.sort_values("visit_index").iloc[0]
    if early.get("morphology_term_hits_in_thyroid_clauses", 0) > 0 and (
        latest.get("negative_term_hits_in_thyroid_clauses", 0) > 0 or latest.get("negative_thyroid_cue_count", 0) > 0
    ):
        score += 2
    if int(group.iloc[0].get("n_unique_fp_seeds", 0)) >= 3:
        score += 1
    return score


def review_focus(group: pd.DataFrame) -> str:
    focuses: List[str] = []
    if (group["thyroid_positive_and_negative_overlap"] > 0).any():
        focuses.append("thyroid_positive_negative_overlap")
    if (group["benign_nodule_mimic_suspected"] > 0).any():
        focuses.append("benign_nodule_mimic")
    if (group["non_thyroid_morphology_source_suspected"] > 0).any():
        focuses.append("non_thyroid_morphology_source")
    latest = group.sort_values("visit_index").iloc[-1]
    early = group.sort_values("visit_index").iloc[0]
    if early.get("morphology_term_hits_in_thyroid_clauses", 0) > 0 and (
        latest.get("negative_term_hits_in_thyroid_clauses", 0) > 0 or latest.get("negative_thyroid_cue_count", 0) > 0
    ):
        focuses.append("historical_positive_latest_negative_conflict")
    if not focuses:
        focuses.append("case_level_manual_review")
    return ";".join(focuses)


def flag_summary(patient_frame: pd.DataFrame) -> pd.DataFrame:
    if patient_frame.empty:
        return pd.DataFrame()
    flags = [
        "any_thyroid_morphology_hit",
        "any_non_thyroid_morphology_hit",
        "any_thyroid_positive_negative_overlap",
        "any_benign_nodule_mimic",
        "any_diffuse_ht_like_signal",
        "latest_visit_has_thyroid_morphology_hit",
        "latest_visit_has_negative_thyroid_cue",
        "early_positive_latest_negative_conflict",
    ]
    rows = []
    n = len(patient_frame)
    for flag in flags:
        count = int((patient_frame[flag].astype(float) > 0).sum())
        rows.append({"flag": flag, "n_patients": count, "fraction_of_fp_patients": count / max(n, 1)})
    focus_counts = patient_frame["recommended_manual_review_focus"].str.get_dummies(sep=";").sum().sort_values(ascending=False)
    for focus, count in focus_counts.items():
        rows.append({"flag": f"focus:{focus}", "n_patients": int(count), "fraction_of_fp_patients": int(count) / max(n, 1)})
    return pd.DataFrame(rows)


def write_report(patient_frame: pd.DataFrame, visit_frame: pd.DataFrame, summary: pd.DataFrame, out_dir: Path) -> str:
    if patient_frame.empty:
        recommendation = "DATA_AUDIT_INCOMPLETE"
        lines = ["# Phase C10 False-Positive Report Source Audit", "", "No patient rows were available."]
        (out_dir / "phase_c10_final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return recommendation
    overlap = int((patient_frame["any_thyroid_positive_negative_overlap"] > 0).sum())
    benign = int((patient_frame["any_benign_nodule_mimic"] > 0).sum())
    latest_negative = int((patient_frame["latest_visit_has_negative_thyroid_cue"] > 0).sum())
    diffuse = int((patient_frame["any_diffuse_ht_like_signal"] > 0).sum())
    if benign >= overlap and benign >= 10:
        recommendation = "AUDIT_BENIGN_NODULE_AND_SECTION_FILTER_BEFORE_TRAINING"
    elif overlap >= 10:
        recommendation = "AUDIT_TEMPORAL_EVIDENCE_CONFLICT_BEFORE_TRAINING"
    else:
        recommendation = "CONTINUE_MANUAL_CASE_REVIEW"
    top_cols = [
        "patient_id",
        "n_visits_parsed",
        "max_pred_prob",
        "n_unique_fp_seeds",
        "any_thyroid_positive_negative_overlap",
        "any_benign_nodule_mimic",
        "latest_visit_has_negative_thyroid_cue",
        "early_positive_latest_negative_conflict",
        "recommended_manual_review_focus",
        "source_audit_priority",
    ]
    top = patient_frame[[col for col in top_cols if col in patient_frame.columns]].head(20)
    visit_cols = [
        "patient_id",
        "visit_date",
        "morphology_term_hits_in_thyroid_clauses",
        "negative_term_hits_in_thyroid_clauses",
        "ht_like_thyroid_cue_count",
        "benign_nodule_thyroid_cue_count",
        "negative_thyroid_cue_count",
        "thyroid_positive_and_negative_overlap",
        "benign_nodule_mimic_suspected",
        "thyroid_text_preview",
    ]
    visit_top = visit_frame[visit_frame["patient_id"].isin(top["patient_id"].head(8))][[col for col in visit_cols if col in visit_frame.columns]].head(30)
    lines = [
        "# Phase C10 False-Positive Report Source Audit",
        "",
        "Phase C10 is analysis-only. No model, data loader, label, split, manifest, or training code was changed.",
        "",
        "## Patient-Level Source Summary",
        "",
        f"- FP patients audited: {len(patient_frame)}.",
        f"- Patients with thyroid positive/negative overlap: {overlap}.",
        f"- Patients with benign/nodular mimic signal: {benign}.",
        f"- Patients whose latest visit has negative thyroid cues: {latest_negative}.",
        f"- Patients with diffuse HT-like signal anywhere: {diffuse}.",
        "",
        "## Flag Summary",
        "",
        frame_to_markdown(summary),
        "",
        "## Highest-Priority Patient Source Review",
        "",
        frame_to_markdown(top),
        "",
        "## Visit-Level Evidence Snippets",
        "",
        frame_to_markdown(visit_top),
        "",
        "## Interpretation",
        "",
        "- Many strict MVP false positives appear tied to mixed longitudinal evidence rather than a simple model-only error.",
        "- Benign thyroid nodule language and low/uneven echo terms can resemble HT morphology while still belonging to label-negative patients.",
        "- Latest-visit negative cues and historical positive cues should be reviewed before changing the classifier.",
        "- Shortcut fields remain audit-only; this report does not justify feeding visit count, image count, or report length into a model.",
        "",
        "## Recommendation",
        "",
        f"`{recommendation}`.",
        "",
        "Before any training pilot, define a report-construction or evidence-filtering hypothesis that can be audited without labels changing and without shortcut variables entering the classifier.",
    ]
    (out_dir / "phase_c10_final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return recommendation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit visit/report sources of strict MVP false positives.")
    parser.add_argument("--c9-patient-audit", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    c9 = read_patient_audit(Path(args.c9_patient_audit))
    manifest = read_manifest_frame(Path(args.manifest))
    visits = visit_rows(c9, manifest)
    patients = patient_summary(visits)
    summary = flag_summary(patients)
    visits.to_csv(out_dir / "c10_fp_visit_source_audit_val.csv", index=False)
    patients.to_csv(out_dir / "c10_fp_patient_source_summary_val.csv", index=False)
    summary.to_csv(out_dir / "c10_fp_source_flag_summary_val.csv", index=False)
    recommendation = write_report(patients, visits, summary, out_dir)
    print(f"Wrote Phase C10 report source audit to {out_dir}")
    print(f"Recommendation: {recommendation}")


if __name__ == "__main__":
    main()
