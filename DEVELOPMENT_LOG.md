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

## 2026-07-03 Distmatch Design Before Edits

### Correction

- The earlier `manifest_matched_bins.jsonl` was coarse patient-level bin matching after manifest construction.
- The intended distmatch must happen during data construction at the visit-history level.

### Planned Distmatch Semantics

1. Build per-patient visit histories from `/label/patient_id/date/*.jpg` and `all_patients.xlsx`.
2. Preserve the original patient-level split from an existing base manifest when available.
3. Work independently inside each split.
4. Use `label=1` selected visit-count distribution as the empirical reference.
5. Keep all valid `label=1` historical visits.
6. For each `label=0` patient, sample a target visit count from the `label=1` empirical distribution, capped by that patient's available visits.
7. Sample the selected negative visits from that patient's history, then sort by time.
8. For every selected visit, use exactly the same image policy for both labels:
   - keep at most `max_images_per_visit`,
   - randomly sample if there are more,
   - repeat-pad if there are fewer but at least one image,
   - record `used_images`, `image_padding_count`, and selected visit metadata only for audit/logging.
9. Do not change labels, task definition, or patient split.

### Validation Plan

- Generate a distmatch manifest on the server.
- Audit `n_visits` and `n_images` shortcut proxy AUC after construction.
- Compare against the previous original and coarse matched manifests.

## 2026-07-03 Distmatch Actual Changes

### Added

- Added `scripts/build_distmatch_manifest.py`.
- Added `configs/dmea_ht_distmatch.yaml`.
- Extended shortcut logging/audit fields with `selected_n_visits`, `raw_n_visits`, `used_images`, `raw_n_images`, and `image_padding_count`.
- Added `--fields` to `scripts/audit_shortcut_distribution.py` so selected-structure shortcut risk can be audited separately from raw audit-only structure.

### Distmatch Behavior

- Preserves existing patient-level split from `manifest.jsonl`.
- Uses split-local `label=1` visit-count distribution as the target distribution.
- Keeps all valid historical visits for `label=1`.
- Resamples `label=0` visit counts from the split-local positive empirical distribution, capped by each patient's available history.
- Applies the same per-visit image policy to both labels: sample up to `max_images_per_visit`, repeat-pad when fewer images exist, and keep audit-only counts in the manifest.
- Updated distmatch to follow the DecAlign5090 data-construction behavior more closely: default `history_cutoff` is now `final_year` rather than dropping only the latest visit. `final_visit` and `none` remain available for ablation.
- Fixed distmatch bio leakage: `bio_values` and `bio_missing_mask` are now derived only from rows whose dates remain after the history cutoff, rather than from the patient's latest overall table row.

## 2026-07-03 Structural Match Design Before Edits

### Motivation

- DecAlign-style distmatch now controls visit/image structure to chance-level shortcut AUC.
- The remaining selected-structure proxy signal comes mainly from bio missingness and report length, which are collection/text-structure shortcuts rather than medical evidence.

### Planned Structural Match

1. Start from `manifest_distmatch_final_year.jsonl`, not the raw manifest.
2. Preserve patient-level split and labels.
3. Inside each split, build a joint key from:
   - `selected_n_visits`,
   - `used_images`,
   - `image_padding_count`,
   - `has_bio`,
   - `bio_missing_count`,
   - split-local quantile bin of `report_length`.
4. For each key, keep the same number of positive and negative patients.
5. Emit `manifest_distmatch_structmatch.jsonl` for strict shortcut-control experiments.

### Actual Changes

- Added `scripts/match_manifest_structural_bins.py`.
- Added `configs/dmea_ht_distmatch_structmatch.yaml`.

### Server Validation

- Server run directory: `/home/linruixin/chen/project/DMEA-HT/runs/dmea_ht_distmatch_structmatch_auc_20260703_145921`.
- Data root: `/data/csb/DMEA-HT/HT_2025.12_25`.
- Manifest: `/data/csb/DMEA-HT/HT_2025.12_25/manifest_distmatch_structmatch.jsonl`.
- Strict structural manifest size: 780 patients.
- Split/label counts:
  - train: 301 positive, 301 negative.
  - val: 47 positive, 47 negative.
  - test: 42 positive, 42 negative.
- Shortcut audit after strict structural matching:
  - selected structural proxy AUC: 0.4967.
  - visit/image-only proxy AUC: 0.3922.
  - bio/report-only proxy AUC: 0.5387.
- Three-seed formal training selected checkpoints by `val_AUC`.
- Validation AUC: 0.7581 +/- 0.0171.
- Test AUC: 0.7729 +/- 0.0363.
- Per-seed test AUC:
  - seed 0: 0.8039.
  - seed 42: 0.7330.
  - seed 3407: 0.7817.

## 2026-07-03 DMEA-v2 Evidence Weak Labels Design Before Edits

### Motivation

- Move the project from generic multimodal fusion toward explicit medical evidence roles.
- Start with a low-risk Phase A that only augments the strict structural manifest with weak evidence labels.
- Do not change the patient-level HT task, labels, splits, image paths, report text, or bio values.

### Planned Changes

- Add `scripts/build_evidence_weak_labels.py`.
- Add `scripts/inspect_manifest_evidence_labels.py`.
- Preserve every original manifest field and append evidence weak-label fields.
- Generate report-derived text labels from fixed dictionaries for morphology, negative, uncertain, and diagnosis-hint evidence.
- Generate `image_morphology_weak_label` only from report morphology/negative weak labels.
- Generate `bio_missing_label` from `bio_missing_mask` and available bio metadata.
- Keep bio abnormal labels conservative: use trusted abnormal flags only when explicitly requested; otherwise write `-1` when reference ranges are unavailable.
- Generate `discordance_state_label` from text/bio evidence states.

### Non-Negotiable Constraints

- Do not rewrite `label`, `split`, or `patient_id`.
- Do not use shortcut fields to derive labels.
- Do not feed shortcut fields into a classifier.
- Do not touch test for model selection.
- Formal training remains three seeds: `0`, `42`, and `3407`.

### Validation Plan

- Run static compile checks for the new scripts.
- Build evidence labels on the server from `manifest_distmatch_structmatch.jsonl`.
- Inspect label distributions by split and label.
- Record the generated evidence manifest path and distribution summary before any v2 model training.

## 2026-07-03 DMEA-v2 Evidence Weak Labels Actual Changes

### Added

- Added `scripts/build_evidence_weak_labels.py`.
- Added `scripts/inspect_manifest_evidence_labels.py`.

### Modified

- No model forward path was changed.
- No training loss was changed.
- No patient labels, splits, image paths, report text, or bio values were changed.

### Validation Results

- Local static check passed:
  - `python -m py_compile scripts/build_evidence_weak_labels.py scripts/inspect_manifest_evidence_labels.py`
- Server static check passed in `/home/linruixin/chen/conda/envs/ma`.
- Built evidence manifest:
  - `/data/csb/DMEA-HT/HT_2025.12_25/manifest_distmatch_structmatch_evidence.jsonl`
- Rows written: 780.
- Overall evidence label distribution:
  - `txt_morphology_label`: 683 positive, 97 negative.
  - `txt_negative_label`: 687 positive, 93 negative.
  - `txt_uncertain_label`: 0 positive, 780 negative.
  - `txt_diag_hint_label`: 0 positive, 780 negative.
  - `bio_immune_abnormal_label`: 780 unknown.
  - `bio_function_abnormal_label`: 780 unknown.
  - `bio_missing_label`: 780 non-missing.
  - `image_morphology_weak_label`: 683 positive, 85 negative, 12 unknown.
  - `discordance_state_label`: 780 uncertain_or_insufficient.
- Split distributions were inspected for train/val/test and retained the strict structural manifest sizes: train 602, val 94, test 84.
- Test was not used for model selection or tuning.

### Remaining Issues

- Current manifest does not provide trusted bio abnormal/reference-range information, so immune/function weak labels are conservatively set to `-1`.
- Discordance state is currently uninformative because bio abnormal evidence is unknown.
- The next v2 training pass should initially enable text/image evidence weak supervision only, or first add trustworthy bio reference-range/abnormal flag derivation.

## 2026-07-03 DMEA-v2 Evidence Weak Labels Phase B

### Motivation

- Audit and refine text/image weak evidence labels before enabling v2 evidence-supervised training.
- Current bio abnormal labels are unavailable, so bio evidence and discordance losses remain disabled.
- The Phase A distributions showed overly broad text morphology/negative positives and many morphology-negative overlaps.

### Planned Changes

- Add `scripts/audit_evidence_label_quality.py`.
- Refine `scripts/build_evidence_weak_labels.py` with negation-aware morphology span matching.
- Separate strong HT-relevant negative phrases from weak local negative findings.
- Add top-level matched term lists and weak-label confidence fields.
- Extend `scripts/inspect_manifest_evidence_labels.py` with joint counts, confidence summaries, top terms, and unknown-label summaries.

### Constraints

- Do not modify `dmea_ht/models.py`, `train.py`, or `dmea_ht/data.py`.
- Do not start model training in this phase.
- Keep `patient_id`, `split`, `label`, image paths, report text, bio values, and bio missing masks unchanged.
- Keep bio immune/function abnormal labels as `-1` unless trusted reference-range or abnormal flag information exists.

### Actual Changes

- Added `scripts/audit_evidence_label_quality.py`.
- Refined `scripts/build_evidence_weak_labels.py`:
  - added `--negation-window`;
  - added negation-aware morphology matching;
  - separated strong HT-relevant negative terms from weak local negative terms;
  - added `matched_*_terms` top-level fields;
  - added weak-label confidence fields.
- Extended `scripts/inspect_manifest_evidence_labels.py` with:
  - overall/split/class distributions;
  - morphology-negative joint counts;
  - confidence summaries;
  - top matched terms;
  - unknown-label summaries.

### Validation Results

- Local static compile passed.
- Server static compile passed in `/home/linruixin/chen/conda/envs/ma`.
- Server repository synced to commit `100ca8c`.
- Rebuilt evidence manifest:
  - `/data/csb/DMEA-HT/HT_2025.12_25/manifest_distmatch_structmatch_evidence_v2.jsonl`
- Inspection report:
  - `/data/csb/DMEA-HT/HT_2025.12_25/evidence_label_inspect_v2.txt`
- Audit CSV:
  - `/data/csb/DMEA-HT/HT_2025.12_25/evidence_label_audit_samples_v2.csv`
- Rows written: 780.
- Updated overall evidence label distribution:
  - `txt_morphology_label`: 680 positive, 100 negative.
  - `txt_negative_label`: 311 strong negative, 469 non-strong-negative.
  - `txt_uncertain_label`: 0 positive, 780 negative.
  - `txt_diag_hint_label`: 0 positive, 780 negative.
  - `bio_immune_abnormal_label`: 780 unknown.
  - `bio_function_abnormal_label`: 780 unknown.
  - `bio_missing_label`: 780 non-missing.
  - `image_morphology_weak_label`: 680 positive, 14 negative, 86 unknown.
  - `discordance_state_label`: 780 uncertain_or_insufficient.
- Morphology/negative joint counts:
  - `morph1_neg0`: 383.
  - `morph0_neg1`: 14.
  - `morph1_neg1`: 297.
  - `morph0_neg0`: 86.
- Mean confidence:
  - `txt_morphology_confidence`: 0.7414.
  - `txt_negative_confidence`: 0.4872.
  - `image_morphology_weak_confidence`: 0.7594.
- Audit CSV exported 238 rows with per-split sampling.
- Test split was inspected only for weak-label distribution quality; no model selection or training was performed.

### Remaining Issues

- `morph1_neg1` remains common, likely because multi-visit reports can contain both positive morphology and negative/normal phrases across visits or findings.
- Strong negative labels should be manually reviewed in the audit CSV before enabling `L_text_negative`.
- Bio immune/function abnormal labels remain unknown, so bio evidence loss and discordance supervision remain disabled.
- No Phase C model training was started.

## 2026-07-06 DMEA-v2 Phase C1 Text/Image Evidence Training

### Motivation

- Connect refined Phase B text/image morphology weak labels to training.
- Keep bio and discordance supervision disabled because bio abnormal labels remain unknown.
- Test whether evidence-role supervision improves patient-level HT prediction without reintroducing shortcut dependence.

### Planned Changes

- Update dataset loading for morphology evidence labels and confidence fields.
- Add text/image morphology auxiliary heads.
- Add confidence-weighted evidence BCE losses with ignore-label support.
- Add Phase C1 configs for text-only evidence and text+image evidence.
- Run syntax checks and server-side smoke/formal validation.

### Constraints

- Do not change task labels, patient splits, image paths, report text, or bio values.
- Do not feed shortcut variables into classifier.
- Do not enable negative, bio, discordance, counterfactual, or matched contrastive losses.
- Select checkpoints by validation AUC only.

### Actual Changes

- Added morphology evidence fields to `PatientHTDataset` with safe defaults for old manifests.
- Added confidence-weighted BCE evidence loss with ignore support for `-1` labels.
- Added optional text and image morphology auxiliary heads.
- Added Phase C1 configs:
  - `configs/dmea_ht_v2_text_morphology_only.yaml`;
  - `configs/dmea_ht_v2_text_image_evidence.yaml`;
  - `configs/dmea_ht_v2_text_image_evidence_smoke.yaml`.
- Updated training/evaluation metrics to log evidence-head diagnostics.

### Validation Results

- Local static compile passed.
- Server static compile passed in `/home/linruixin/chen/conda/envs/ma`.
- Server synthetic forward/loss check passed.
- Server smoke run completed:
  - `/home/linruixin/chen/project/DMEA-HT/runs/dmea_ht_v2_text_image_evidence_smoke_20260706`.
- Formal text morphology only run completed:
  - `/home/linruixin/chen/project/DMEA-HT/runs/dmea_ht_v2_text_morphology_only_20260706`.
  - validation AUC: 0.7782 +/- 0.0350.
  - validation AUPRC: 0.7872 +/- 0.0503.
  - validation F1: 0.6140 +/- 0.1058.
  - validation sensitivity: 0.5177 +/- 0.1597.
  - validation specificity: 0.8582 +/- 0.0747.
  - validation balanced accuracy: 0.6879 +/- 0.0430.
  - test AUC: 0.7819 +/- 0.0148.
- Formal text + image morphology run completed:
  - `/home/linruixin/chen/project/DMEA-HT/runs/dmea_ht_v2_text_image_evidence_20260706`.
  - validation AUC: 0.7691 +/- 0.0223.
  - validation AUPRC: 0.7743 +/- 0.0323.
  - validation F1: 0.6287 +/- 0.0119.
  - validation sensitivity: 0.5461 +/- 0.0123.
  - validation specificity: 0.8085 +/- 0.0369.
  - validation balanced accuracy: 0.6773 +/- 0.0163.
  - test AUC: 0.7927 +/- 0.0199.
- Relative to the strict MVP validation AUC of 0.7581 +/- 0.0171, both C1 variants remained competitive; text morphology only gave the best validation AUC.
- Test metrics are recorded for reporting only and were not used for model selection.

### Remaining Issues

- Shortcut audit with the selected-structure field set remained chance-level:
  - fields: `selected_n_visits`, `used_images`, `image_padding_count`, `has_bio`, `bio_missing_count`, `report_length`.
  - shortcut-only AUC: 0.4967.
  - shortcut-only AUPRC: 0.5150.
- Prediction/shortcut Spearman correlations were generally modest. The largest observed absolute correlations were around 0.35 for `n_images`/`n_visits` in some test seeds and around 0.31 for `report_length`.
- The default shortcut audit field set includes raw audit-only fields such as `raw_n_visits` and `raw_n_images`; that all-field manifest proxy is not the strict selected-structure control criterion.
- `text_morphology_auc` is zero in the diagnostic CSV because valid text morphology labels are single-class in validation/test, so AUC is not defined and the metric helper safely returns 0.
- `image_morphology_auc` is diagnostic only and should not drive checkpoint selection.

## 2026-07-06 DMEA-v2 Phase C2 Text Evidence Anchor Refinement

### Motivation

- C1 showed text morphology supervision improves validation AUC over the strict MVP.
- Text+image morphology did not improve validation AUC over text-only, so image BCE supervision remains disabled.
- C2 upgrades text morphology supervision from an auxiliary head into a fused Text Evidence Anchor.

### Planned Changes

- Add `TextEvidenceAnchor`.
- Optionally fuse `text_morphology_anchor` into patient-level fusion.
- Add C2 weight-scan configs.
- Add validation-derived threshold analysis.
- Extend evidence diagnostics.
- Run selected-structure shortcut audit after formal runs.

### Constraints

- Do not change labels, splits, image paths, report text, or bio values.
- Do not feed shortcut variables into classifier.
- Do not enable negative, bio, discordance, counterfactual, matched SupCon, or image BCE losses.
- Select by validation AUC only.

### Actual Changes

- Added `TextEvidenceAnchor` with query attention over text tokens.
- Added config switches:
  - `use_text_evidence_anchor`;
  - `fuse_text_morphology_anchor`.
- Fused `text_morphology_anchor` into patient-level fusion only when enabled by config.
- Extended prediction CSV diagnostics with:
  - `pred_prob`;
  - `txt_morphology_label`;
  - `txt_morphology_confidence`;
  - `matched_morphology_terms`;
  - `text_morphology_prob`;
  - `text_morphology_anchor_norm`;
  - `text_morphology_anchor_mean`.
- Added validation-derived threshold script:
  - `scripts/evaluate_thresholds.py`.
- Added C2 configs:
  - `configs/dmea_ht_v2_c2_text_anchor_w001.yaml`;
  - `configs/dmea_ht_v2_c2_text_anchor_w003.yaml`;
  - `configs/dmea_ht_v2_c2_text_anchor_w005.yaml`;
  - `configs/dmea_ht_v2_c2_text_anchor_w010.yaml`;
  - `configs/dmea_ht_v2_c2_text_anchor_smoke.yaml`.

### Validation Results

- Local static compile passed.
- Server static compile passed in `/home/linruixin/chen/conda/envs/ma`.
- Server C2 synthetic forward check passed.
- Server C2 smoke run completed:
  - `/home/linruixin/chen/project/DMEA-HT/runs/dmea_ht_v2_c2_text_anchor_smoke_20260706`.
- Threshold analysis smoke check completed:
  - `/home/linruixin/chen/project/DMEA-HT/runs/dmea_ht_v2_c2_text_anchor_smoke_20260706/thresholds/seed_0`.
- Smoke prediction CSV contains anchor diagnostics and matched morphology term fields.

### Remaining Issues

- Formal C2 weight scan still needs to complete for weights 0.01, 0.03, 0.05, and 0.10.
- Shortcut audit and threshold reports must be run after each formal C2 run.
