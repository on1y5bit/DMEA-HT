# Frozen C13 Model Card

## Model Identity

- Route: `C13_TEMPORAL_FOCUS_DMEA_HT`
- Config: `configs/dmea_ht_v2_c13_temporal_focus_stress_seeds.yaml`
- Manifest: `/data/csb/DMEA-HT/HT_2025.12_25/manifest_distmatch_structmatch_evidence_v2_c13_temporal_focus.jsonl`
- Formal seeds: `[0, 42, 3407]`
- Primary checkpoint-selection metric: validation AUC
- Delivery status: `PASS`

The frozen delivery is the C13 route with three independently trained, validation-selected single-model checkpoints. The formal performance claim is the three-seed mean; no test-selected seed or ensemble is claimed.

## Validation Performance

- AUC: `0.8665 +/- 0.0077`
- AUPRC: `0.8570 +/- 0.0049`
- Sensitivity: `0.6525 +/- 0.1568`
- Specificity: `0.8511 +/- 0.0426`

Test AUC `0.8460 +/- 0.0077` is reporting-only and was not used for selection.

## Shortcut Safety

- Validation pooled max absolute Spearman: `0.1549`
- Validation pooled shortcut linear R2: `0.0601`
- Validation shortcut-only label AUC, audit-only: `0.4762`

Shortcut and audit fields are never classifier inputs.

## Known Limitations

- The validation AUC target of 0.90 was not reached.
- Sensitivity is seed-sensitive.
- C14-C found concentrated hard-patient inversion structure.
- C14-E found insufficient matched controls and no generalizable correction mechanism; route: `DATA_LIMIT_NO_GENERAL_MODEL_FIX`.
- C15 training remains blocked. C13 is frozen as the current strict best and the limitation must be reported.
