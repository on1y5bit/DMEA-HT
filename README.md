# DMEA-HT

DMEA-HT is a patient-level multimodal Hashimoto thyroiditis prediction scaffold. It is designed around evidence-aware fusion and shortcut auditing rather than image/text/bio feature concatenation alone.

## Frozen Strict-Best Route

The final strict-best route under the completed C13-C14E evidence gates is:

- route: `C13_TEMPORAL_FOCUS_DMEA_HT`;
- config: `configs/dmea_ht_v2_c13_temporal_focus_stress_seeds.yaml`;
- manifest: `/data/csb/DMEA-HT/HT_2025.12_25/manifest_distmatch_structmatch_evidence_v2_c13_temporal_focus.jsonl`;
- formal seeds: `[0, 42, 3407]`;
- validation AUC: `0.8665 +/- 0.0077`;
- validation AUPRC: `0.8570 +/- 0.0049`.

The formal claim is the three-seed mean of independently trained, validation-AUC-selected single-model checkpoints. Test metrics are reporting-only; no test-selected seed or ensemble is claimed.

C14-E concluded `DATA_LIMIT_NO_GENERAL_MODEL_FIX`. C15 remains blocked, C13 is frozen as the current strict best, and the validation AUC 0.90 target was not reached.

See [FINAL_MODEL_SELECTION.md](FINAL_MODEL_SELECTION.md) and `analysis_reports/final_c13_delivery/` for the reproducibility and artifact-verification package.

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

The command above is a scaffold example, not the frozen C13 reproduction command. The exact C13 command and checkpoint paths are recorded in `analysis_reports/final_c13_delivery/final_reproducibility_commands.md` after server verification.

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
