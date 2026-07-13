# C19 Main Merge Verification

## Status

`MAIN_CONTAINS_CANONICAL_C17_C18`

## Git

- Canonical server project: `/home/linruixin/chen/project/DMEA-HT`.
- Active branch after merge: `main`.
- Verified HEAD: `136982cd04be995d9a294a4f7ab367494e491ca3`.
- Merge method: fast-forward from the existing canonical feature branch.
- No new GitHub branch or worktree was created.

## Preserved C17 Artifacts

- Checkpoints: `seed_0_best.pt`, `seed_42_best.pt`, and `seed_3407_best.pt`.
- Validation predictions: three seed-specific CSV files.
- Reporting-only test predictions: three seed-specific CSV files.
- Formal metrics, positive-preservation audit, pairwise inversion audit, shortcut audit, formal gate, and final report are present under `analysis_reports/phase_c17_dema/`.
- Reference validation AUC remains `0.8696242644 +/- 0.0074797246`.

## Preserved C18 Artifacts

- C18-D and C18-DH run metrics by seed and epoch are present under their canonical run directories.
- C18 formal metrics, merged route audits, transition analysis, gate JSON, and final report are present under `analysis_reports/phase_c18_dema/`.
- C18-D validation AUC remains `0.8741511996 +/- 0.0102532835`.
- C18-DH validation AUC remains `0.8717368342 +/- 0.0073739500`.
- C18 final decision remains `DEMA_C18_NEGATIVE_INFLATION`; selected route remains C17 because of positive suppression and negative inflation.

## C19 Entry State

- C17 remains the strict-best model.
- C19 may use the frozen promoted C17 checkpoint and evidence representation only after the C19-A polarity audit and server static/synthetic gate pass.
