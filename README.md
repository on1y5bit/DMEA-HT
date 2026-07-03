# DMEA-HT

DMEA-HT is a patient-level multimodal Hashimoto thyroiditis prediction scaffold. It is designed around evidence-aware fusion and shortcut auditing rather than image/text/bio feature concatenation alone.

## Server Contract

- Data root: `/data/csb/DMEA-HT/HT_2025.12_25`
- Conda env: `/home/linruixin/chen/conda/envs/ma`
- Primary metric: `val_AUC`
- Formal seeds: `0, 42, 3407`

## Quick Start

Prepare a patient-level manifest as JSONL or CSV. Each row should include:

- `patient_id`
- `label`
- optional `split` with `train`, `val`, or `test`
- optional image paths in `image_paths`, `images`, or `image_path`
- optional text in `report_text`, `text`, or token IDs in `report_input_ids`
- optional bio features in `bio_values`
- audit-only shortcut fields such as `n_images`, `n_visits`, `has_bio`, `bio_missing_count`, `report_length`, `source_folder`

Run training:

```bash
python train.py --config configs/dmea_ht_mvp.yaml
```

Run shortcut audit:

```bash
python scripts/audit_shortcut_distribution.py \
  --manifest /data/csb/DMEA-HT/HT_2025.12_25/manifest.jsonl \
  --out-dir runs/audit_manifest
```

Run server MVP:

```bash
bash scripts/run_server_mvp.sh
```

## Outputs

Formal runs write:

- `reports/metrics_by_seed.csv`
- `reports/metrics_summary.csv`
- `reports/confusion_matrix_by_seed.csv`
- `reports/shortcut_audit.csv`
- `reports/shortcut_proxy_auc.csv`
- `predictions/val_predictions_seed_*.csv`
- `predictions/test_predictions_seed_*.csv`
- `checkpoints/seed_*_best.pt`

Shortcut fields are logged for audit, but they are not classifier inputs.

