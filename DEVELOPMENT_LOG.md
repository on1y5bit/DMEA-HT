# DMEA-HT Development Log

## 2026-07-03 Design Before Edits

### Context

- Local workspace: `C:\Users\bang\Desktop\project\DMEA-HT`.
- GitHub remote: `https://github.com/on1y5bit/DMEA-HT.git`.
- Server data root: `/data/csb/DMEA-HT/HT_2025.12_25`.
- Server runtime environment: `/home/linruixin/chen/conda/envs/ma`.
- Primary model-selection metric for this project: validation AUC.

### Non-Negotiable Constraints

- Keep the task as patient-level binary HT prediction.
- Keep all modalities for the same `patient_id` in one split.
- Do not derive or rewrite labels from shortcut variables.
- Do not feed `n_images`, `n_visits`, `has_bio`, `bio_missing_count`, `report_length`, `source_folder`, or similar shortcut variables into the classifier.
- Use shortcut variables only for audit, matching, sample weighting, and logging.
- Formal seeds are `0`, `42`, and `3407`; report mean and standard deviation, not only the best seed.
- Do not tune on test output.

### Planned MVP

1. Create a patient-level manifest-driven dataset that supports image, text, and bio fields plus audit-only shortcut fields.
2. Add a minimal DMEA-HT model:
   - image encoder with patient-level image masking,
   - text encoder,
   - bio encoder with missing-mask separation,
   - patient-anchor fusion,
   - evidence-role and discordance outputs,
   - evidence-conservation classifier.
3. Add baseline support for image-only, text-only, bio-only, concat, and DMEA fusion variants.
4. Add AUC-first training with secondary binary metrics.
5. Add shortcut audit and shortcut-only proxy AUC reports.
6. Add bin-based matching and sample reweighting utilities.
7. Add server-oriented scripts and documentation for local edit -> GitHub push -> server pull/run.

### Validation Plan

- Local validation is limited to static syntax checks because the local machine may not have the server ML stack.
- Runtime validation should run on the server under `/home/linruixin/chen/conda/envs/ma`.

## 2026-07-03 Actual Changes

### Added

- Added `train.py` with AUC-first patient-level training across seeds `0`, `42`, and `3407`.
- Added `dmea_ht/data.py` for manifest-driven patient-level loading, fixed-K image masking, report tokenization, bio values, missing masks, sample weights, and audit-only shortcut fields.
- Added `dmea_ht/models.py` with the first runnable DMEA-HT MVP:
  - image encoder,
  - text encoder,
  - bio encoder with medical/observation separation,
  - evidence-role alignment,
  - patient-anchor fusion,
  - discordance features,
  - evidence-conservation classifier,
  - image-only/text-only/bio-only/concat baselines.
- Added `dmea_ht/metrics.py` for AUC, AUPRC, ACC, F1, sensitivity, specificity, precision, recall, balanced accuracy, and confusion matrix counts.
- Added `data_matching/` utilities for shortcut binning, distribution reports, bin matching, sample reweighting, propensity scoring, and stratified patient split.
- Added scripts:
  - `scripts/audit_shortcut_distribution.py`
  - `scripts/validate_shortcut.py`
  - `scripts/counterfactual_eval.py`
  - `scripts/apply_sample_reweighting.py`
  - `scripts/match_manifest_by_bins.py`
  - `scripts/build_manifest_from_table.py`
  - `scripts/build_manifest_from_dmea_layout.py`
  - `scripts/inspect_table.py`
  - `scripts/run_server_mvp.sh`
- Added `configs/dmea_ht_smoke.yaml` for a one-seed, one-epoch server smoke run.
- Added `configs/dmea_ht_mvp.yaml`, `README.md`, and `requirements.txt`.

### Minimal Local Checks

- Ran Python syntax compilation for all new Python files.
- Did not run local model training; this workspace is for coding/push, and runtime validation belongs on the server environment.

### Remaining Issues

- The GitHub repository was empty at clone time, so this is the first project scaffold.
- The server rejected non-interactive SSH because no key-based auth is available in this local environment. Password-based interactive SSH may need to be completed by the user, or a deploy key/session helper must be added.
- The actual server data layout still needs live inspection. If `/data/csb/DMEA-HT/HT_2025.12_25/manifest.jsonl` does not exist, use `scripts/build_manifest_from_table.py` or adapt a small manifest builder after inspecting the data files.
