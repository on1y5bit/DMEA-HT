from __future__ import annotations

import ast
import csv
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


SHORTCUT_FIELDS = [
    "n_images",
    "n_visits",
    "selected_n_visits",
    "raw_n_visits",
    "used_images",
    "raw_n_images",
    "has_bio",
    "bio_missing_count",
    "report_length",
    "source_folder",
    "image_padding_count",
    "padding_count",
]


def read_manifest(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        with path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            for key in ("samples", "records", "patients"):
                if key in data:
                    return list(data[key])
        return list(data)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_maybe_list(value: Any) -> List[Any]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if stripped[0:1] in "[{":
            try:
                parsed = ast.literal_eval(stripped)
                return parsed if isinstance(parsed, list) else [parsed]
            except (ValueError, SyntaxError):
                pass
        if "|" in stripped:
            return [item for item in stripped.split("|") if item]
        if ";" in stripped:
            return [item for item in stripped.split(";") if item]
        return [stripped]
    return [value]


def stable_token_id(token: str, vocab_size: int) -> int:
    digest = hashlib.md5(token.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % max(vocab_size - 2, 1) + 2


def tokenize_text(text: str, max_length: int, vocab_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = list(text.strip())
    ids = [1] + [stable_token_id(token, vocab_size) for token in tokens[: max_length - 2]] + [1]
    ids = ids[:max_length]
    mask = [1] * len(ids)
    if len(ids) < max_length:
        pad = max_length - len(ids)
        ids.extend([0] * pad)
        mask.extend([0] * pad)
    return torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.long)


def patient_split(rows: Sequence[Dict[str, Any]], seed: int = 42, val_frac: float = 0.15, test_frac: float = 0.15) -> List[str]:
    patients: Dict[str, int] = {}
    for row in rows:
        patients[str(row["patient_id"])] = int(row["label"])

    by_label: Dict[int, List[str]] = {0: [], 1: []}
    for pid, label in patients.items():
        by_label.setdefault(label, []).append(pid)

    rng = random.Random(seed)
    split_by_patient: Dict[str, str] = {}
    for label, pids in by_label.items():
        rng.shuffle(pids)
        n = len(pids)
        n_test = max(1, round(n * test_frac)) if n >= 3 else 0
        n_val = max(1, round(n * val_frac)) if n >= 3 else 0
        for pid in pids[:n_test]:
            split_by_patient[pid] = "test"
        for pid in pids[n_test : n_test + n_val]:
            split_by_patient[pid] = "val"
        for pid in pids[n_test + n_val :]:
            split_by_patient[pid] = "train"

    return [split_by_patient[str(row["patient_id"])] for row in rows]


class PatientHTDataset(Dataset):
    def __init__(
        self,
        rows: Sequence[Dict[str, Any]],
        data_root: str | Path,
        split: str,
        max_images: int,
        image_size: int,
        text_max_length: int,
        text_vocab_size: int,
        bio_dim: int,
    ) -> None:
        self.rows = [row for row in rows if str(row.get("split", "")).lower() == split]
        self.data_root = Path(data_root)
        self.split = split
        self.max_images = max_images
        self.image_size = image_size
        self.text_max_length = text_max_length
        self.text_vocab_size = text_vocab_size
        self.bio_dim = bio_dim

    def __len__(self) -> int:
        return len(self.rows)

    def _load_image_tensor(self, path: str) -> torch.Tensor:
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

    def _image_paths(self, row: Dict[str, Any]) -> List[str]:
        for key in ("image_paths", "images", "image_path", "image"):
            paths = parse_maybe_list(row.get(key))
            if paths:
                return [str(path) for path in paths]
        return []

    def _bio_values(self, row: Dict[str, Any]) -> torch.Tensor:
        values = parse_maybe_list(row.get("bio_values"))
        numeric: List[float] = []
        for value in values:
            try:
                numeric.append(float(value))
            except (TypeError, ValueError):
                numeric.append(0.0)
        numeric = numeric[: self.bio_dim]
        if len(numeric) < self.bio_dim:
            numeric.extend([0.0] * (self.bio_dim - len(numeric)))
        return torch.tensor(numeric, dtype=torch.float32)

    def _bio_missing_mask(self, row: Dict[str, Any], values: torch.Tensor) -> torch.Tensor:
        mask_values = parse_maybe_list(row.get("bio_missing_mask"))
        if mask_values:
            mask = [float(v) for v in mask_values[: self.bio_dim]]
            if len(mask) < self.bio_dim:
                mask.extend([1.0] * (self.bio_dim - len(mask)))
            return torch.tensor(mask, dtype=torch.float32)
        return (values == 0).float()

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.rows[index]
        paths = self._image_paths(row)
        selected_paths = paths[: self.max_images]
        images = [self._load_image_tensor(path) for path in selected_paths]
        image_mask = [1.0] * len(images)
        while len(images) < self.max_images:
            images.append(torch.zeros(3, self.image_size, self.image_size, dtype=torch.float32))
            image_mask.append(0.0)

        text = str(row.get("report_text") or row.get("text") or "")
        token_ids, attention_mask = tokenize_text(text, self.text_max_length, self.text_vocab_size)
        bio_values = self._bio_values(row)
        bio_missing_mask = self._bio_missing_mask(row, bio_values)
        bio_abnormal_flags = torch.tensor(
            [float(v) for v in parse_maybe_list(row.get("bio_abnormal_flags"))[: self.bio_dim]]
            or [0.0] * self.bio_dim,
            dtype=torch.float32,
        )
        if bio_abnormal_flags.numel() < self.bio_dim:
            bio_abnormal_flags = torch.cat([bio_abnormal_flags, torch.zeros(self.bio_dim - bio_abnormal_flags.numel())])

        shortcuts = {field: row.get(field, "") for field in SHORTCUT_FIELDS}
        shortcuts["n_images"] = row.get("n_images", len(paths))
        shortcuts["has_bio"] = row.get("has_bio", int(float(bio_missing_mask.sum()) < self.bio_dim))
        shortcuts["bio_missing_count"] = row.get("bio_missing_count", int(float(bio_missing_mask.sum())))
        shortcuts["report_length"] = row.get("report_length", len(text))
        shortcuts["padding_count"] = row.get("padding_count", max(self.max_images - len(selected_paths), 0))

        return {
            "patient_id": str(row["patient_id"]),
            "label": torch.tensor(float(row["label"]), dtype=torch.float32),
            "images": torch.stack(images, dim=0),
            "image_mask": torch.tensor(image_mask, dtype=torch.float32),
            "report_input_ids": token_ids,
            "report_attention_mask": attention_mask,
            "bio_values": bio_values,
            "bio_missing_mask": bio_missing_mask,
            "bio_abnormal_flags": bio_abnormal_flags,
            "sample_weight": torch.tensor(float(row.get("sample_weight", 1.0)), dtype=torch.float32),
            "shortcuts": shortcuts,
        }


def collate_patient_batch(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    tensor_keys = [
        "label",
        "images",
        "image_mask",
        "report_input_ids",
        "report_attention_mask",
        "bio_values",
        "bio_missing_mask",
        "bio_abnormal_flags",
        "sample_weight",
    ]
    out = {key: torch.stack([item[key] for item in batch]) for key in tensor_keys}
    out["patient_id"] = [item["patient_id"] for item in batch]
    out["shortcuts"] = [item["shortcuts"] for item in batch]
    return out
