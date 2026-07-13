# Mistaken C16 DSSA Revert Report

## Decision

The prior Phase C16 DSSA/shared-specific plan is invalid for this project. The work is stopped, excluded from model selection and reporting, and reverted on its unmerged feature branch without rewriting history.

Final status label:

```text
MISGUIDED_C16_STOPPED_AND_REVERTED
```

## Revert Method

- Applicable case: unmerged feature-branch commits (Case B).
- Verified pre-C16 base: `b91bd1d`.
- Mainline status before correction: both `main` and `origin/main` pointed to `b91bd1d`.
- Full committed delta preserved in `analysis_reports/c16_correction/mistaken_c16_changes.patch`.
- `dmea_ht/models.py` and `train.py` were restored exactly from `b91bd1d`.
- Files created only for C16 DSSA were removed from the corrected branch state.
- Historical development-log entries were preserved and followed by an explicit correction.
- No `reset --hard`, clean, force push, stash, or history rewrite was used.

## Reverted Files

Restored to the exact pre-C16 contents:

- `dmea_ht/models.py`
- `train.py`

Removed because they existed only for the mistaken DSSA plan:

- `dmea_ht/alignment.py`
- `configs/dmea_ht_v2_c16_dssa_smoke.yaml`
- `configs/dmea_ht_v2_c16_dssa_seed0_pilot.yaml`
- `configs/dmea_ht_v2_c16_dssa_stress_seeds.yaml`
- `scripts/audit_phase_c16_alignment_health.py`
- `scripts/collect_phase_c16_report.py`
- `analysis_reports/phase_c16/c16_synthetic_smoke.json`
- `analysis_reports/phase_c16/c16_synthetic_smoke_checks.csv`

## Training Disposition

- A seed-0 two-epoch smoke had completed before correction.
- The full seed-0 pilot and stress seeds were never started.
- No matching C16 process was alive at correction time.
- The smoke checkpoint and reports remain on the isolated server worktree only as audit evidence.
- The server artifacts are marked `ABORTED_MISGUIDED_C16`, `NOT_FOR_MODEL_SELECTION`, and `NOT_FOR_REPORTING`.
- No C16 smoke metric is a valid project result.

## Protected Work

The correction does not alter the C13 temporal-focus manifest, model checkpoints, predictions, final delivery package, C14-A through C14-E analysis, patient labels, patient-level splits, or user-created/untracked files. C13 remains the current strict-best baseline.

## Corrected Direction

The next permitted phase is C16-MEA: Disease-Mechanism and Evidence-Aware Alignment. It starts with a design audit only. No C16-MEA implementation or training is authorized until that audit is complete and reviewed.
