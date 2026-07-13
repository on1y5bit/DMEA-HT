# C13 Final Reproducibility Commands

Run from `/home/linruixin/chen/project/DMEA-HT` with the frozen environment:

```bash
PY=/home/linruixin/chen/conda/envs/ma/bin/python
MANIFEST=/data/csb/DMEA-HT/HT_2025.12_25/manifest_distmatch_structmatch_evidence_v2_c13_temporal_focus.jsonl
RUN_DIR=runs/dmea_ht_v2_c13_temporal_focus_stress_seeds
CONFIG=configs/dmea_ht_v2_c13_temporal_focus_stress_seeds.yaml
```

The original formal training command was:

```bash
$PY train.py --config $CONFIG
```

This command retrains the three formal seeds `[0, 42, 3407]`; it is recorded for reproducibility and is not authorized as a new experiment.

Verify the saved C13 checkpoints and inference contract without training:

```bash
$PY scripts/collect_phase_c14b_report.py \
  --manifest $MANIFEST \
  --run-dir $RUN_DIR \
  --output-dir analysis_reports/phase_c14b \
  --device auto --batch-size 4 --seeds 0,42,3407
```

Regenerate the final delivery inventory:

```bash
$PY scripts/collect_final_c13_delivery.py \
  --run-dir $RUN_DIR \
  --config $CONFIG \
  --manifest $MANIFEST \
  --output-dir analysis_reports/final_c13_delivery
```

Formal checkpoint paths:

- `runs/dmea_ht_v2_c13_temporal_focus_stress_seeds/checkpoints/seed_0_best.pt`
- `runs/dmea_ht_v2_c13_temporal_focus_stress_seeds/checkpoints/seed_42_best.pt`
- `runs/dmea_ht_v2_c13_temporal_focus_stress_seeds/checkpoints/seed_3407_best.pt`

Checkpoint selection used validation AUC only. Test metrics are reporting-only. Shortcut fields are audit-only and are not classifier inputs.
