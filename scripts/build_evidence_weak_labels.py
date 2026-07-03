from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dmea_ht.data import parse_maybe_list, read_manifest


MORPHOLOGY_POSITIVE_TERMS = [
    "回声不均",
    "回声欠均",
    "回声减低",
    "低回声",
    "弥漫性改变",
    "弥漫性病变",
    "实质回声粗糙",
    "实质回声不均",
    "桥本样改变",
    "血流丰富",
    "血供丰富",
]

NEGATIVE_TERMS = [
    "未见明显异常",
    "未见异常",
    "回声均匀",
    "未见明显弥漫性改变",
    "未见明显桥本表现",
    "未见明显异常血流",
]

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
    "桥本",
    "桥本甲状腺炎",
    "HT",
    "慢性淋巴细胞性甲状腺炎",
]

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


def normalize_name(name: str) -> str:
    return re.sub(r"[\s_/\\()（）-]+", "", str(name)).lower()


def matched_terms(text: str, terms: Iterable[str]) -> List[str]:
    matches: List[str] = []
    upper_text = text.upper()
    for term in terms:
        if term.upper() == "HT":
            if re.search(r"(?<![A-Z])HT(?![A-Z])", upper_text):
                matches.append(term)
        elif term and term in text:
            matches.append(term)
    return matches


def binary_from_matches(matches: Sequence[str]) -> int:
    return 1 if matches else 0


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
    indices = []
    for idx, column in enumerate(columns):
        if normalize_name(column) in aliases:
            indices.append(idx)
    return indices


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
    if not observed:
        return -1
    if not trusted_flags:
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
    if txt_negative == 1 and txt_morphology == 0:
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


def add_evidence_labels(row: Dict[str, Any], bio_columns: Sequence[str], trust_abnormal_flags: bool) -> Dict[str, Any]:
    out = dict(row)
    text = str(row.get("report_text") or row.get("text") or row.get("report") or "")
    morphology_terms = matched_terms(text, MORPHOLOGY_POSITIVE_TERMS)
    negative_terms = matched_terms(text, NEGATIVE_TERMS)
    uncertain_terms = matched_terms(text, UNCERTAIN_TERMS)
    diag_hint_terms = matched_terms(text, DIAG_HINT_TERMS)

    txt_morphology = binary_from_matches(morphology_terms)
    txt_negative = binary_from_matches(negative_terms)
    txt_uncertain = binary_from_matches(uncertain_terms)
    txt_diag_hint = binary_from_matches(diag_hint_terms)

    missing = int_list(row.get("bio_missing_mask"))
    flags = int_list(row.get("bio_abnormal_flags"))
    trusted_flags = row_has_trusted_abnormal_flags(row, trust_abnormal_flags)
    immune_indices = group_indices(bio_columns, IMMUNE_ALIASES)
    function_indices = group_indices(bio_columns, FUNCTION_ALIASES)
    bio_immune = group_abnormal_label(flags, missing, immune_indices, trusted_flags)
    bio_function = group_abnormal_label(flags, missing, function_indices, trusted_flags)
    bio_missing = bio_missing_label(row, bio_columns)
    image_morphology = derive_image_morphology_label(txt_morphology, txt_negative)
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
            "evidence_label_source": {
                "text_terms": {
                    "morphology": morphology_terms,
                    "negative": negative_terms,
                    "uncertain": uncertain_terms,
                    "diag_hint": diag_hint_terms,
                },
                "bio_columns": list(bio_columns),
                "bio_immune_indices": list(immune_indices),
                "bio_function_indices": list(function_indices),
                "bio_abnormal_source": "trusted_flags" if trusted_flags else "unknown_no_reference_range",
                "rules_version": "v2_phase_a_2026_07_03",
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
    out_rows = [add_evidence_labels(row, bio_columns, args.trust_bio_abnormal_flags) for row in rows]
    write_jsonl(out_rows, Path(args.output))
    print(f"wrote {len(out_rows)} rows to {args.output}")
    for field in EVIDENCE_LABEL_FIELDS:
        counts: Dict[Any, int] = {}
        for row in out_rows:
            counts[row[field]] = counts.get(row[field], 0) + 1
        print(f"{field}: {dict(sorted(counts.items(), key=lambda item: item[0]))}")


if __name__ == "__main__":
    main()
