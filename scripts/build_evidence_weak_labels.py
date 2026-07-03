from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dmea_ht.data import parse_maybe_list, read_manifest


MORPHOLOGY_HIGH_CONF = [
    "弥漫性改变",
    "弥漫性病变",
    "桥本样改变",
    "实质回声不均",
    "实质回声粗糙",
    "腺体回声不均",
    "弥漫性回声改变",
]
MORPHOLOGY_MED_CONF = [
    "回声不均",
    "回声欠均",
    "回声欠均匀",
    "回声减低",
    "低回声",
]
MORPHOLOGY_POSITIVE_TERMS = MORPHOLOGY_HIGH_CONF + MORPHOLOGY_MED_CONF

NEGATIVE_STRONG_TERMS = [
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
]
NEGATIVE_WEAK_TERMS = [
    "未见明显结节",
    "未见结节",
    "未见明显肿大淋巴结",
    "未见明显钙化",
    "未见异常血流",
    "未见明显异常血流",
]
NEGATIVE_TERMS = NEGATIVE_STRONG_TERMS + NEGATIVE_WEAK_TERMS

UNCERTAIN_TERMS = [
    "考虑",
    "可疑",
    "倾向",
    "建议结合",
    "建议复查",
    "不能除外",
    "待排",
]

DIAG_HINT_TERMS = [
    "桥本甲状腺炎",
    "慢性淋巴细胞性甲状腺炎",
    "桥本",
    "HT",
]

NEGATION_TRIGGERS = [
    "未见明显",
    "无明显",
    "未见明确",
    "未发现",
    "未提示",
    "未显示",
    "未见",
    "没有",
    "否认",
    "无",
]

UNCERTAIN_CONTEXT_TERMS = ["HT", "桥本", "弥漫性改变", "弥漫性病变", "回声不均", "回声减低"]

DEFAULT_BIO_COLUMNS = ["sex", "age", "TgAb", "FT3", "FT4", "TPOAb", "TSH"]
IMMUNE_ALIASES = {
    "tpoab",
    "tgab",
    "trab",
    "甲状腺过氧化物酶抗体",
    "甲状腺球蛋白抗体",
    "促甲状腺素受体抗体",
}
FUNCTION_ALIASES = {"tsh", "ft3", "ft4", "t3", "t4"}

EVIDENCE_LABEL_FIELDS = [
    "txt_morphology_label",
    "txt_negative_label",
    "txt_uncertain_label",
    "txt_diag_hint_label",
    "bio_immune_abnormal_label",
    "bio_function_abnormal_label",
    "bio_missing_label",
    "image_morphology_weak_label",
    "discordance_state_label",
]
CONFIDENCE_FIELDS = [
    "txt_morphology_confidence",
    "txt_negative_confidence",
    "txt_uncertain_confidence",
    "txt_diag_hint_confidence",
    "image_morphology_weak_confidence",
]


def unique_in_order(items: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


def normalize_name(name: str) -> str:
    return re.sub(r"[\s_/\\()（）-]+", "", str(name)).lower()


def find_occurrences(text: str, term: str) -> List[int]:
    if not term:
        return []
    if term.upper() == "HT":
        return [match.start() for match in re.finditer(r"(?<![A-Z])HT(?![A-Z])", text.upper())]
    starts: List[int] = []
    start = 0
    while True:
        idx = text.find(term, start)
        if idx < 0:
            break
        starts.append(idx)
        start = idx + max(len(term), 1)
    return starts


def has_negation_before(text: str, start: int, window: int) -> bool:
    before = text[max(0, start - window) : start]
    return any(trigger in before for trigger in NEGATION_TRIGGERS)


def match_morphology_terms(text: str, negation_window: int) -> Tuple[List[str], List[str]]:
    positive: List[str] = []
    negated: List[str] = []
    for term in MORPHOLOGY_POSITIVE_TERMS:
        for start in find_occurrences(text, term):
            if has_negation_before(text, start, negation_window):
                negated.append(term)
            else:
                positive.append(term)
    return unique_in_order(positive), unique_in_order(negated)


def match_terms(text: str, terms: Sequence[str]) -> List[str]:
    return unique_in_order(term for term in terms if find_occurrences(text, term))


def confidence_from_terms(matches: Sequence[str], high_terms: Sequence[str], medium_terms: Sequence[str]) -> float:
    if any(term in high_terms for term in matches):
        return 1.0
    if any(term in medium_terms for term in matches):
        return 0.7
    return 0.3 if matches else 0.0


def negative_confidence(strong_matches: Sequence[str], weak_matches: Sequence[str]) -> float:
    if strong_matches:
        return 1.0
    if weak_matches:
        return 0.5
    return 0.0


def uncertain_confidence(text: str, matches: Sequence[str]) -> float:
    if not matches:
        return 0.0
    for term in matches:
        for start in find_occurrences(text, term):
            window = text[max(0, start - 12) : start + len(term) + 12]
            if any(context in window.upper() if context == "HT" else context in window for context in UNCERTAIN_CONTEXT_TERMS):
                return 1.0
    return 0.5


def diagnosis_confidence(text: str, matches: Sequence[str]) -> float:
    if not matches:
        return 0.0
    for term in matches:
        for start in find_occurrences(text, term):
            window = text[max(0, start - 8) : start + len(term) + 8]
            if any(marker in window for marker in ("考虑", "可能", "倾向", "可疑")):
                return 0.7
    return 1.0


def numeric_list(value: Any) -> List[float]:
    out: List[float] = []
    for item in parse_maybe_list(value):
        try:
            out.append(float(item))
        except (TypeError, ValueError):
            out.append(0.0)
    return out


def int_list(value: Any) -> List[int]:
    out: List[int] = []
    for item in parse_maybe_list(value):
        try:
            out.append(int(float(item)))
        except (TypeError, ValueError):
            out.append(1)
    return out


def group_indices(columns: Sequence[str], aliases: set[str]) -> List[int]:
    return [idx for idx, column in enumerate(columns) if normalize_name(column) in aliases]


def row_has_trusted_abnormal_flags(row: Dict[str, Any], trust_abnormal_flags: bool) -> bool:
    if trust_abnormal_flags:
        return bool(parse_maybe_list(row.get("bio_abnormal_flags")))
    for key in ("bio_abnormal_flags_trusted", "bio_abnormal_source", "bio_reference_range_source"):
        value = row.get(key)
        if value not in (None, "", 0, "0", False, "false", "False"):
            return bool(parse_maybe_list(row.get("bio_abnormal_flags")))
    return False


def group_abnormal_label(
    flags: Sequence[int],
    missing: Sequence[int],
    indices: Sequence[int],
    trusted_flags: bool,
) -> int:
    if not indices:
        return -1
    observed = [idx for idx in indices if idx < len(missing) and int(missing[idx]) == 0]
    if not observed or not trusted_flags:
        return -1
    observed_flags = [int(flags[idx]) for idx in observed if idx < len(flags)]
    if not observed_flags:
        return -1
    return 1 if any(flag == 1 for flag in observed_flags) else 0


def bio_missing_label(row: Dict[str, Any], bio_columns: Sequence[str]) -> int:
    missing = int_list(row.get("bio_missing_mask"))
    values = numeric_list(row.get("bio_values"))
    if missing:
        key_indices = group_indices(bio_columns, IMMUNE_ALIASES | FUNCTION_ALIASES)
        if not key_indices:
            key_indices = list(range(len(missing)))
        observed_key_count = sum(1 for idx in key_indices if idx < len(missing) and int(missing[idx]) == 0)
        return 1 if observed_key_count == 0 else 0
    if values:
        return 0
    has_bio = row.get("has_bio")
    if has_bio not in (None, ""):
        try:
            return 0 if int(float(has_bio)) == 1 else 1
        except (TypeError, ValueError):
            pass
    return 1


def derive_image_morphology_label(txt_morphology: int, txt_negative: int) -> int:
    if txt_morphology == 1:
        return 1
    if txt_negative == 1:
        return 0
    return -1


def derive_discordance_state(
    txt_morphology: int,
    txt_negative: int,
    txt_diag_hint: int,
    bio_immune: int,
    bio_function: int,
    bio_missing: int,
) -> int:
    if bio_immune == -1 or bio_function == -1:
        return 5
    morph_pos = txt_morphology == 1 or txt_diag_hint == 1
    morph_neg = txt_negative == 1 and not morph_pos
    bio_pos = bio_immune == 1 or bio_function == 1
    bio_neg = bio_immune == 0 and bio_function == 0
    if morph_pos and bio_pos:
        return 1
    if morph_neg and bio_neg:
        return 0
    if morph_pos and bio_neg:
        return 2
    if bio_pos and morph_neg:
        return 3
    if morph_pos and bio_missing == 1:
        return 4
    return 5


def add_evidence_labels(
    row: Dict[str, Any],
    bio_columns: Sequence[str],
    trust_abnormal_flags: bool,
    negation_window: int,
) -> Dict[str, Any]:
    out = dict(row)
    text = str(row.get("report_text") or row.get("text") or row.get("report") or "")
    morphology_terms, negated_morphology_terms = match_morphology_terms(text, negation_window)
    strong_negative_terms = match_terms(text, NEGATIVE_STRONG_TERMS)
    weak_negative_terms = match_terms(text, NEGATIVE_WEAK_TERMS)
    negative_terms = unique_in_order(list(strong_negative_terms) + list(weak_negative_terms))
    uncertain_terms = match_terms(text, UNCERTAIN_TERMS)
    diag_hint_terms = match_terms(text, DIAG_HINT_TERMS)

    txt_morphology = 1 if morphology_terms else 0
    txt_negative = 1 if strong_negative_terms else 0
    txt_uncertain = 1 if uncertain_terms else 0
    txt_diag_hint = 1 if diag_hint_terms else 0

    txt_morphology_conf = confidence_from_terms(morphology_terms, MORPHOLOGY_HIGH_CONF, MORPHOLOGY_MED_CONF)
    txt_negative_conf = negative_confidence(strong_negative_terms, weak_negative_terms)
    txt_uncertain_conf = uncertain_confidence(text, uncertain_terms)
    txt_diag_hint_conf = diagnosis_confidence(text, diag_hint_terms)

    missing = int_list(row.get("bio_missing_mask"))
    flags = int_list(row.get("bio_abnormal_flags"))
    trusted_flags = row_has_trusted_abnormal_flags(row, trust_abnormal_flags)
    immune_indices = group_indices(bio_columns, IMMUNE_ALIASES)
    function_indices = group_indices(bio_columns, FUNCTION_ALIASES)
    bio_immune = group_abnormal_label(flags, missing, immune_indices, trusted_flags)
    bio_function = group_abnormal_label(flags, missing, function_indices, trusted_flags)
    bio_missing = bio_missing_label(row, bio_columns)
    image_morphology = derive_image_morphology_label(txt_morphology, txt_negative)
    if image_morphology == 1:
        image_morphology_conf = txt_morphology_conf
    elif image_morphology == 0:
        image_morphology_conf = txt_negative_conf
    else:
        image_morphology_conf = 0.0
    discordance_state = derive_discordance_state(
        txt_morphology,
        txt_negative,
        txt_diag_hint,
        bio_immune,
        bio_function,
        bio_missing,
    )

    out.update(
        {
            "txt_morphology_label": txt_morphology,
            "txt_negative_label": txt_negative,
            "txt_uncertain_label": txt_uncertain,
            "txt_diag_hint_label": txt_diag_hint,
            "bio_immune_abnormal_label": bio_immune,
            "bio_function_abnormal_label": bio_function,
            "bio_missing_label": bio_missing,
            "image_morphology_weak_label": image_morphology,
            "discordance_state_label": discordance_state,
            "matched_morphology_terms": morphology_terms,
            "matched_negative_terms": negative_terms,
            "matched_uncertain_terms": uncertain_terms,
            "matched_diag_hint_terms": diag_hint_terms,
            "txt_morphology_confidence": txt_morphology_conf,
            "txt_negative_confidence": txt_negative_conf,
            "txt_uncertain_confidence": txt_uncertain_conf,
            "txt_diag_hint_confidence": txt_diag_hint_conf,
            "image_morphology_weak_confidence": image_morphology_conf,
            "evidence_label_source": {
                "rules_version": "v2_phase_b_2026_07_03",
                "negation_window": negation_window,
                "negated_morphology_terms": negated_morphology_terms,
                "matched_strong_negative_terms": strong_negative_terms,
                "matched_weak_negative_terms": weak_negative_terms,
                "bio_abnormal_source": "trusted_flags" if trusted_flags else "unknown_no_reference_range",
            },
        }
    )
    return out


def write_jsonl(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Append DMEA-HT v2 evidence weak labels to a manifest.")
    parser.add_argument("--input", required=True, help="Input manifest JSON/JSONL/CSV path.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--negation-window", type=int, default=10, help="Characters before morphology term checked for negation.")
    parser.add_argument(
        "--bio-columns",
        default=",".join(DEFAULT_BIO_COLUMNS),
        help="Comma-separated order of bio_values columns.",
    )
    parser.add_argument(
        "--trust-bio-abnormal-flags",
        action="store_true",
        help="Trust existing bio_abnormal_flags as true abnormal labels.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bio_columns = [part.strip() for part in str(args.bio_columns).split(",") if part.strip()]
    rows = read_manifest(args.input)
    out_rows = [
        add_evidence_labels(row, bio_columns, args.trust_bio_abnormal_flags, int(args.negation_window))
        for row in rows
    ]
    write_jsonl(out_rows, Path(args.output))
    print(f"wrote {len(out_rows)} rows to {args.output}")
    for field in EVIDENCE_LABEL_FIELDS:
        counts: Dict[Any, int] = {}
        for row in out_rows:
            counts[row[field]] = counts.get(row[field], 0) + 1
        print(f"{field}: {dict(sorted(counts.items(), key=lambda item: item[0]))}")
    for field in CONFIDENCE_FIELDS:
        values = [float(row.get(field, 0.0)) for row in out_rows]
        mean = sum(values) / len(values) if values else 0.0
        print(f"{field}_mean: {mean:.4f}")


if __name__ == "__main__":
    main()
