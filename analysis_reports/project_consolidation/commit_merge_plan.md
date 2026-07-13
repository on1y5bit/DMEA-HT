# DEMA-HT Project Consolidation Commit Plan

## Canonical Starting State

- Canonical server directory: `/home/linruixin/chen/project/DMEA-HT`
- Existing canonical branch: `feature/c16-disease-state-shared-specific-alignment`
- Server canonical worktree HEAD at inventory time: `e69f1af74cad3ac6617cdca616d1d96e423bc538`
- `main` remains the pre-C17 line at `b91bd1d` and does not contain the promoted C17 implementation.
- The existing feature branch is retained because it is the current development line and already contains the corrected C16-MEA history after the reverted DSSA attempt.

## Verified Source

- Target source commit: `93983877c864cbb0c10234360596b985ae32862b`
- C17 implementation commit: `fc3154a0d5bfcc426af590ed8292352d9f041ce0`
- C17 formal audit collector commit: `eef23384a5a4405798ce3bcfb17e017eea75ea3d`
- The C17 formal decision log is included in the target source commit.

## Ancestry And Operation

- `3363e98` is an ancestor of `9398387`.
- `eef2338` is an ancestor of `9398387`.
- The server canonical HEAD `e69f1af` is an ancestor of the target C17 line.
- The planned operation is a fast-forward of the existing canonical feature branch to the verified C17 source line.
- No cherry-pick is required. No new GitHub branch or Git worktree is required.
- No reset, clean, force push, or overwrite operation is permitted.

## Artifact Operation

- Preserve the one-time inventory at `/home/linruixin/chen/project_archive/dema_ht_consolidation_20260713_205355/`.
- Migrate C17 formal training and formal audit artifacts into canonical `runs/` and `analysis_reports/` using conflict-safe, ignore-existing copying.
- Same-path hash conflicts must be copied under `analysis_reports/project_consolidation/conflicts/` and reported; canonical files must never be silently overwritten.
- C17 reproduction must pass before old worktrees or directories are removed.

## Finalization

- The final canonical commit, artifact conflict count, retained archive, branch cleanup, and worktree cleanup status will be appended after server verification.
