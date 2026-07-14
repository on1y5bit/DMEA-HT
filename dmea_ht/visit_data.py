from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from dmea_ht.data import SHORTCUT_FIELDS, tokenize_text
from dmea_ht.mechanism_evidence_alignment import TEXT_MASK_KEYS, build_text_evidence_masks


BIO_COLUMNS = ("sex", "age", "TgAb", "FT3", "FT4", "TPOAb", "TSH")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def is_missing(value: Any) -> bool:
    if value is None or value == "":
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def normalize_patient_id(value: Any) -> str:
    if is_missing(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


def normalize_date(value: Any) -> str:
    if is_missing(value):
        return ""
    parsed = pd.to_datetime(value, errors="coerce")
    if not pd.isna(parsed):
        return parsed.strftime("%Y-%m-%d")
    return str(value).strip()[:10]


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class SourceVisit:
    patient_id: str
    visit_date: str
    source_row_ids: Tuple[int, ...]
    report_text: str
    bio_values: Tuple[float, ...]
    bio_missing_mask: Tuple[int, ...]
    bio_source_row_id: int | None


def _bio_from_row(row: Mapping[str, Any]) -> Tuple[Tuple[float, ...], Tuple[int, ...]]:
    values: List[float] = []
    missing: List[int] = []
    for column in BIO_COLUMNS:
        value = row.get(column)
        if is_missing(value):
            values.append(0.0)
            missing.append(1)
            continue
        try:
            values.append(float(value))
            missing.append(0)
        except (TypeError, ValueError):
            values.append(0.0)
            missing.append(1)
    return tuple(values), tuple(missing)


def load_source_visits(data_root: str | Path) -> Dict[Tuple[str, str], SourceVisit]:
    table = pd.read_excel(Path(data_root) / "all_patients.xlsx")
    table = table.rename(columns={"patient_Id": "patient_id"}).copy()
    required = {"patient_id", "time", "report", *BIO_COLUMNS}
    missing_columns = sorted(required - set(table.columns))
    if missing_columns:
        raise RuntimeError(f"Missing all_patients.xlsx columns: {missing_columns}")
    table["patient_id"] = table["patient_id"].map(normalize_patient_id)
    table["_date"] = table["time"].map(normalize_date)
    table["_source_row_id"] = list(range(2, len(table) + 2))

    visits: Dict[Tuple[str, str], SourceVisit] = {}
    for (patient_id, visit_date), group in table.groupby(["patient_id", "_date"], sort=False):
        ordered = group.sort_values("_source_row_id", kind="stable")
        report_parts = [str(value).strip() for value in ordered["report"] if not is_missing(value)]
        bio_row = ordered.iloc[-1]
        values, missing = _bio_from_row(bio_row)
        visits[(str(patient_id), str(visit_date))] = SourceVisit(
            patient_id=str(patient_id),
            visit_date=str(visit_date),
            source_row_ids=tuple(int(value) for value in ordered["_source_row_id"]),
            report_text="\n".join(part for part in report_parts if part),
            bio_values=values,
            bio_missing_mask=missing,
            bio_source_row_id=int(bio_row["_source_row_id"]),
        )
    return visits


def image_visit_key(path: str) -> Tuple[str, str, str]:
    parts = PurePosixPath(str(path).replace("\\", "/")).parts
    if len(parts) < 4:
        return "", "", ""
    return str(parts[0]), normalize_patient_id(parts[1]), normalize_date(parts[2])


def group_selected_images(row: Mapping[str, Any]) -> Dict[str, List[str]]:
    label = str(int(float(row["label"])))
    patient_id = normalize_patient_id(row["patient_id"])
    selected_dates = {normalize_date(value) for value in row.get("selected_visit_dates", [])}
    grouped: Dict[str, List[str]] = {date: [] for date in selected_dates}
    for raw_path in row.get("image_paths", []):
        path = str(raw_path).replace("\\", "/")
        path_label, path_patient, path_date = image_visit_key(path)
        if path_label != label or path_patient != patient_id or path_date not in selected_dates:
            raise RuntimeError(
                f"Selected image path does not match patient/visit: patient={patient_id} path={path}"
            )
        grouped[path_date].append(path)
    return grouped


def _patient_bio_fallback(row: Mapping[str, Any], visits: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    has_dated_bio = any(
        any(int(value) == 0 for value in visit["bio_missing_mask_if_dated"])
        for visit in visits
        if visit["dated_bio_row_id"] is not None
    )
    values = [float(value) for value in row.get("bio_values", [0.0] * len(BIO_COLUMNS))]
    missing = [int(value) for value in row.get("bio_missing_mask", [1] * len(BIO_COLUMNS))]
    valid = (not has_dated_bio) and any(value == 0 for value in missing)
    return {
        "valid": bool(valid),
        "bio_values": values if valid else [0.0] * len(BIO_COLUMNS),
        "bio_missing_mask": missing if valid else [1] * len(BIO_COLUMNS),
        "source": "c13_pre_cutoff_patient_bio_once" if valid else "not_required",
    }


def reconstruct_patient_row(
    row: Mapping[str, Any],
    source_visits: Mapping[Tuple[str, str], SourceVisit],
) -> Dict[str, Any]:
    patient_id = normalize_patient_id(row["patient_id"])
    indexed_dates = [(index, normalize_date(value)) for index, value in enumerate(row.get("selected_visit_dates", []))]
    if not indexed_dates:
        raise RuntimeError(f"Patient {patient_id} has no selected visit dates")
    if len({date for _, date in indexed_dates}) != len(indexed_dates):
        raise RuntimeError(f"Patient {patient_id} has duplicate selected visit dates")
    ordered_dates = [date for _, date in sorted(indexed_dates, key=lambda item: (item[1], item[0]))]
    grouped_images = group_selected_images(row)

    visits: List[Dict[str, Any]] = []
    for rank, visit_date in enumerate(ordered_dates):
        source = source_visits.get((patient_id, visit_date))
        if source is None:
            report_text = ""
            report_source = "unavailable"
            report_unavailable_reason = "no_matching_all_patients_row"
            source_row_ids: List[int] = []
            bio_values = [0.0] * len(BIO_COLUMNS)
            bio_missing = [1] * len(BIO_COLUMNS)
            bio_row_id = None
            bio_relation = "unavailable"
        else:
            report_text = source.report_text
            report_source = "all_patients.xlsx:report" if report_text else "unavailable"
            report_unavailable_reason = "" if report_text else "source_report_missing"
            source_row_ids = list(source.source_row_ids)
            bio_values = list(source.bio_values)
            bio_missing = list(source.bio_missing_mask)
            bio_row_id = source.bio_source_row_id
            bio_relation = "exact_visit_date"
        visits.append(
            {
                "visit_id": f"{patient_id}:{visit_date}",
                "visit_date": visit_date,
                "visit_rank": rank,
                "image_paths": list(grouped_images.get(visit_date, [])),
                "report_text": report_text,
                "report_source": report_source,
                "report_source_row_ids": source_row_ids,
                "report_unavailable_reason": report_unavailable_reason,
                "dated_bio_row_id": bio_row_id,
                "bio_time_relation": bio_relation,
                "bio_values_if_dated": bio_values,
                "bio_missing_mask_if_dated": bio_missing,
            }
        )

    output = dict(row)
    output["patient_id"] = patient_id
    output["visits"] = visits
    output["patient_bio_fallback"] = _patient_bio_fallback(row, visits)
    output["c27_visit_reconstruction"] = {
        "source": "c13_selected_dates_and_images_plus_all_patients_exact_date",
        "order": "oldest_to_latest_stable_source_order",
        "history_cutoff": "unchanged_from_c13_final_year",
        "test_used_for_rule_design": False,
    }
    return output


def build_visit_manifest(
    base_rows: Sequence[Mapping[str, Any]],
    source_visits: Mapping[Tuple[str, str], SourceVisit],
) -> List[Dict[str, Any]]:
    rows = [reconstruct_patient_row(row, source_visits) for row in base_rows]
    if len({normalize_patient_id(row["patient_id"]) for row in rows}) != len(rows):
        raise RuntimeError("Duplicate patient_id after C27 visit reconstruction")
    return rows


C27_AUDIT_FIELDS = (
    *SHORTCUT_FIELDS,
    "reconstructable_visit_count",
    "visit_report_coverage",
    "dated_bio_visit_count",
)


class VisitPatientDataset(Dataset):
    def __init__(
        self,
        rows: Sequence[Dict[str, Any]],
        data_root: str | Path,
        split: str,
        image_size: int,
        text_max_length: int,
        text_vocab_size: int,
        bio_dim: int,
        max_images_per_visit: int,
    ) -> None:
        self.rows = [row for row in rows if str(row.get("split", "")).lower() == split]
        self.data_root = Path(data_root)
        self.split = split
        self.image_size = int(image_size)
        self.text_max_length = int(text_max_length)
        self.text_vocab_size = int(text_vocab_size)
        self.bio_dim = int(bio_dim)
        self.max_images_per_visit = int(max_images_per_visit)

    def __len__(self) -> int:
        return len(self.rows)

    def _load_image(self, path: str) -> torch.Tensor:
        full_path = Path(path)
        if not full_path.is_absolute():
            full_path = self.data_root / full_path
        try:
            from PIL import Image
            from torchvision import transforms

            image = Image.open(full_path).convert("RGB")
            transform = transforms.Compose(
                [
                    transforms.Resize((self.image_size, self.image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
            return transform(image)
        except Exception:
            return torch.zeros(3, self.image_size, self.image_size, dtype=torch.float32)

    def _visit_images(self, paths: Sequence[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        selected = [str(path) for path in paths[: self.max_images_per_visit]]
        images: List[torch.Tensor] = []
        valid: List[float] = []
        for path in selected:
            images.append(self._load_image(path))
            valid.append(1.0)
        while len(images) < self.max_images_per_visit:
            images.append(torch.zeros(3, self.image_size, self.image_size, dtype=torch.float32))
            valid.append(0.0)
        return torch.stack(images), torch.tensor(valid, dtype=torch.float32)

    def _bio_tensor(self, values: Sequence[Any], missing: Sequence[Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        numeric: List[float] = []
        for value in list(values)[: self.bio_dim]:
            try:
                numeric.append(float(value))
            except (TypeError, ValueError):
                numeric.append(0.0)
        numeric.extend([0.0] * (self.bio_dim - len(numeric)))
        mask = [float(value) for value in list(missing)[: self.bio_dim]]
        mask.extend([1.0] * (self.bio_dim - len(mask)))
        return torch.tensor(numeric, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.rows[index]
        visits = list(row.get("visits", []))
        if not visits:
            raise RuntimeError(f"C27 patient has no visits: {row.get('patient_id')}")
        visit_tensors: List[Dict[str, torch.Tensor]] = []
        report_present: List[bool] = []
        dated_bio_present: List[bool] = []
        source_dates: List[str] = []
        for visit in visits:
            images, image_mask = self._visit_images(visit.get("image_paths", []))
            text = str(visit.get("report_text", "") or "")
            input_ids, attention_mask = tokenize_text(text, self.text_max_length, self.text_vocab_size)
            text_masks = build_text_evidence_masks(text, self.text_max_length)
            bio_values, bio_missing = self._bio_tensor(
                visit.get("bio_values_if_dated", []), visit.get("bio_missing_mask_if_dated", [])
            )
            support_present = bool(
                torch.maximum(text_masks["text_support_mask"], text_masks["text_diagnostic_hint_mask"]).sum() > 0
            )
            opposition_present = bool(text_masks["text_opposition_mask"].sum() > 0)
            visit_tensors.append(
                {
                    "images": images,
                    "image_mask": image_mask,
                    "report_input_ids": input_ids,
                    "report_attention_mask": attention_mask,
                    "bio_values": bio_values,
                    "bio_missing_mask": bio_missing,
                    "bio_abnormal_flags": torch.zeros(self.bio_dim, dtype=torch.float32),
                    "visit_text_valid": torch.tensor(bool(text.strip()), dtype=torch.bool),
                    "visit_support_present": torch.tensor(support_present, dtype=torch.bool),
                    "visit_opposition_present": torch.tensor(opposition_present, dtype=torch.bool),
                    **text_masks,
                }
            )
            report_present.append(bool(text.strip()))
            dated_bio_present.append(visit.get("dated_bio_row_id") is not None)
            source_dates.append(str(visit.get("visit_date", "")))

        fallback = dict(row.get("patient_bio_fallback", {}))
        fallback_values, fallback_missing = self._bio_tensor(
            fallback.get("bio_values", []), fallback.get("bio_missing_mask", [])
        )
        shortcuts = {field: row.get(field, "") for field in SHORTCUT_FIELDS}
        shortcuts.update(
            {
                "reconstructable_visit_count": int(sum(report_present)),
                "visit_report_coverage": float(sum(report_present) / len(visits)),
                "dated_bio_visit_count": int(sum(dated_bio_present)),
            }
        )
        return {
            "patient_id": normalize_patient_id(row["patient_id"]),
            "label": torch.tensor(float(row["label"]), dtype=torch.float32),
            "visits": visit_tensors,
            "visit_dates": source_dates,
            "visit_report_present": report_present,
            "visit_dated_bio_present": dated_bio_present,
            "fallback_bio_values": fallback_values,
            "fallback_bio_missing_mask": fallback_missing,
            "fallback_bio_valid": torch.tensor(bool(fallback.get("valid", False)), dtype=torch.bool),
            "shortcuts": shortcuts,
        }


def _pad_visits(items: Sequence[Dict[str, torch.Tensor]], key: str, max_visits: int) -> torch.Tensor:
    example = items[0][key]
    padded = torch.zeros((max_visits, *example.shape), dtype=example.dtype)
    if key == "bio_missing_mask":
        padded.fill_(1)
    for index, item in enumerate(items):
        padded[index] = item[key]
    return padded


def collate_visit_batch(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    max_visits = max(len(item["visits"]) for item in batch)
    visit_tensor_keys = (
        "images",
        "image_mask",
        "report_input_ids",
        "report_attention_mask",
        *TEXT_MASK_KEYS,
        "bio_values",
        "bio_missing_mask",
        "bio_abnormal_flags",
        "visit_text_valid",
        "visit_support_present",
        "visit_opposition_present",
    )
    output: Dict[str, Any] = {
        key: torch.stack([_pad_visits(item["visits"], key, max_visits) for item in batch])
        for key in visit_tensor_keys
    }
    output["visit_mask"] = torch.stack(
        [
            torch.tensor([True] * len(item["visits"]) + [False] * (max_visits - len(item["visits"])))
            for item in batch
        ]
    )
    output["label"] = torch.stack([item["label"] for item in batch])
    output["fallback_bio_values"] = torch.stack([item["fallback_bio_values"] for item in batch])
    output["fallback_bio_missing_mask"] = torch.stack([item["fallback_bio_missing_mask"] for item in batch])
    output["fallback_bio_valid"] = torch.stack([item["fallback_bio_valid"] for item in batch])
    output["patient_id"] = [item["patient_id"] for item in batch]
    output["visit_dates"] = [item["visit_dates"] for item in batch]
    output["visit_report_present"] = [item["visit_report_present"] for item in batch]
    output["visit_dated_bio_present"] = [item["visit_dated_bio_present"] for item in batch]
    output["shortcuts"] = [item["shortcuts"] for item in batch]
    return output
