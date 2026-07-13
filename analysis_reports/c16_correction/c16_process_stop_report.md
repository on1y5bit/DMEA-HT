# Mistaken C16 Process Stop Report

- Checked at: 2026-07-13 16:15 +08:00
- Server: `linruixin@10.21.71.74`
- Project-scoped worktree: `/home/linruixin/chen/project/DMEA-HT-c16`
- Matching live C16 DSSA training processes: none
- Processes terminated during correction: `0`
- Seed-0 smoke had already exited before the correction.
- Seed-0 full pilot was never launched.
- Stress seeds were never launched.

## Preserved Invalid Outputs

- Run directory: `runs/dmea_ht_v2_c16_dssa_smoke`
- Audit directory: `analysis_reports/phase_c16_smoke`
- Config: `configs/dmea_ht_v2_c16_dssa_smoke.yaml`
- Seed: `0`
- Best epoch: `2`
- Checkpoint: `checkpoints/seed_0_best.pt`

Both server directories contain these marker files:

```text
ABORTED_MISGUIDED_C16
NOT_FOR_MODEL_SELECTION
NOT_FOR_REPORTING
```

The smoke metrics are retained only as audit evidence of the stopped mistake. They must not be compared with C13, used for checkpoint selection, or included as a valid experiment in project reporting.
