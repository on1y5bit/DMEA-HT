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

- Formal C2 weight scan completed for weights 0.01, 0.03, 0.05, and 0.10.
- Formal run directories:
  - `/home/linruixin/chen/project/DMEA-HT/runs/dmea_ht_v2_c2_text_anchor_w001_20260706`
  - `/home/linruixin/chen/project/DMEA-HT/runs/dmea_ht_v2_c2_text_anchor_w003_20260706`
  - `/home/linruixin/chen/project/DMEA-HT/runs/dmea_ht_v2_c2_text_anchor_w005_20260706`
  - `/home/linruixin/chen/project/DMEA-HT/runs/dmea_ht_v2_c2_text_anchor_w010_20260706`
- Comparison table:
  - `/home/linruixin/chen/project/DMEA-HT/runs/dmea_ht_v2_c2_text_anchor_comparison_20260706.csv`
- C2 validation AUC results:
  - w=0.01: 0.7709 +/- 0.0227.
  - w=0.03: 0.7691 +/- 0.0114.
  - w=0.05: 0.7746 +/- 0.0173.
  - w=0.10: 0.7687 +/- 0.0252.
- C2 validation AUPRC results:
  - w=0.01: 0.7757 +/- 0.0273.
  - w=0.03: 0.7711 +/- 0.0328.
  - w=0.05: 0.7710 +/- 0.0309.
  - w=0.10: 0.7774 +/- 0.0265.
- C2 test AUC results, recorded for reporting only:
  - w=0.01: 0.7884 +/- 0.0081.
  - w=0.03: 0.7851 +/- 0.0162.
  - w=0.05: 0.7906 +/- 0.0023.
  - w=0.10: 0.7806 +/- 0.0231.
- Threshold reports were generated for every C2 formal seed under each run's `thresholds/seed_*` directory.
- Selected-structure shortcut audit remained chance-level for every compared run:
  - shortcut-only AUC: 0.4967.
  - shortcut-only AUPRC: 0.5150.
- Maximum absolute prediction/shortcut Spearman correlation:
  - Strict MVP: 0.2478.
  - C1 text morphology only: 0.3529.
  - C1 text + image morphology: 0.3356.
  - C2 w=0.01: 0.3354.
  - C2 w=0.03: 0.3656.
  - C2 w=0.05: 0.3366.
  - C2 w=0.10: 0.3287.
- C2 did not beat C1 text morphology only on validation AUC. By the validation-only selection rule, the current main candidate remains C1 text morphology only.
- Among C2 variants, w=0.05 is the best validation-AUC C2 setting, but it is not promoted over C1.

## 2026-07-06 DMEA-v2 Phase C3 Result Consolidation and Decision Gate

### Plan

- Do not build new model modules or launch new formal training.
- Do not modify `dmea_ht/models.py`, `dmea_ht/data.py`, `train.py`, existing model configs, labels, splits, or data.
- Consolidate strict MVP, C1, and C2 completed runs using validation AUC as the only promotion metric.
- Treat test metrics as reporting-only and never use them for model selection.
- Audit prediction residual association with selected structural shortcut fields:
  - `selected_n_visits`;
  - `used_images`;
  - `image_padding_count`;
  - `has_bio`;
  - `bio_missing_count`;
  - `report_length`.
- Generate Phase C3 reports under `analysis_reports/phase_c3/`.
- Add a reusable decision-gate script and documentation for future candidates.
- Run static compile for all new scripts before server execution.

### Expected Outputs

- `analysis_reports/phase_c3/model_comparison_table.csv`
- `analysis_reports/phase_c3/model_comparison_report.md`
- `analysis_reports/phase_c3/c1_evidence_effects_val.csv`
- `analysis_reports/phase_c3/c1_evidence_effects_test_reporting_only.csv`
- `analysis_reports/phase_c3/c1_evidence_effects_report.md`
- `analysis_reports/phase_c3/shortcut_residual_audit.csv`
- `analysis_reports/phase_c3/shortcut_residual_audit_report.md`
- `analysis_reports/phase_c3/decision_gate.md`
- `analysis_reports/phase_c3/decision_gate_summary.csv`
- `analysis_reports/phase_c3/phase_c3_final_report.md`

### Actual Changes

- Added Phase C3 result consolidation script:
  - `scripts/collect_phase_c3_model_comparison.py`.
- Added strict MVP vs C1 text evidence prediction-delta analysis:
  - `scripts/analyze_c1_evidence_effects.py`.
- Added prediction/shortcut residual audit:
  - `scripts/audit_prediction_shortcut_residual.py`.
- Added reusable decision-gate script:
  - `scripts/apply_decision_gate.py`.
- Added decision-gate documentation:
  - `docs/decision_gate.md`.
- Kept Phase C3 analysis-only:
  - no model module changes;
  - no dataset/manifest construction changes;
  - no `train.py` changes;
  - no existing training config changes;
  - no new formal training launched.

### Validation Results

- Local static compile passed for all four new scripts.
- Server static compile passed in `/home/linruixin/chen/conda/envs/ma`.
- Server GitHub pull was blocked by transient HTTP/TLS transport failures, so the local `main` branch was synced to the server via `DMEA-HT_phasec3.bundle`.
- Server code head after sync:
  - `a1af16a`.
- Server Phase C3 reports generated under:
  - `/home/linruixin/chen/project/DMEA-HT/analysis_reports/phase_c3`.
- Generated report files:
  - `model_comparison_table.csv`;
  - `model_comparison_report.md`;
  - `c1_evidence_effects_val.csv`;
  - `c1_evidence_effects_test_reporting_only.csv`;
  - `c1_evidence_effects_report.md`;
  - `shortcut_residual_audit.csv`;
  - `shortcut_residual_audit_report.md`;
  - `decision_gate.md`;
  - `decision_gate_summary.csv`;
  - `phase_c3_final_report.md`.

### Consolidated Results

- Current main candidate by validation-only rule:
  - C1 text morphology only.
- C1 text morphology only:
  - validation AUC: 0.7782 +/- 0.0350;
  - test AUC, reporting-only: 0.7819 +/- 0.0148;
  - decision gate: `PASS_CURRENT`.
- Strict MVP reference:
  - validation AUC: 0.7581 +/- 0.0171;
  - test AUC, reporting-only: 0.7729 +/- 0.0363;
  - decision gate: `REFERENCE`.
- C1 text + image evidence:
  - validation AUC: 0.7691 +/- 0.0223;
  - decision gate: `FAIL`, because it did not beat current best validation AUC.
- Best C2 variant by validation AUC:
  - C2 text anchor w=0.05;
  - validation AUC: 0.7746 +/- 0.0173;
  - decision gate: `FAIL`, because it did not beat current best validation AUC.
- All other C2 variants also failed promotion under the validation-AUC gate.

### C1 Evidence Effect Summary

- Validation split:
  - 282 patient-seed prediction rows;
  - mean C1-MVP probability delta: -0.2014;
  - mean C1-MVP absolute-error delta: -0.0080, where negative is better.
- Test split, reporting-only:
  - 252 patient-seed prediction rows;
  - mean C1-MVP probability delta: -0.1785;
  - mean C1-MVP absolute-error delta: 0.0024.

### Shortcut Residual Gate Summary

- Decision-gate summary now includes pooled validation `max_abs_prediction_shortcut_spearman`.
- Pooled validation residual Spearman values:
  - strict MVP: 0.1737;
  - C1 text morphology only: 0.1796;
  - C1 text + image evidence: 0.1593;
  - C2 w=0.01: 0.1749;
  - C2 w=0.03: 0.1685;
  - C2 w=0.05: 0.1650;
  - C2 w=0.10: 0.1309.
- Pandas emitted constant-input Spearman warnings for constant shortcut columns during server audit; this is expected for fields with no within-split variance and does not stop report generation.

### Decision

- No new improvement is claimed in Phase C3.
- The current main candidate remains C1 text morphology only.
- Future training should not start directly from a new idea. A candidate should first pass the documented pilot gate:
  - unchanged patient-level split and task definition;
  - static compile;
  - validation-AUC improvement over the current main candidate;
  - shortcut residual audit without a new structural shortcut concern;
  - test metrics used only for reporting.

## 2026-07-06 DMEA-v2 Phase C4 C1 Stability and Pilot Weight Sweep

### Plan

- Keep Phase C4 narrow: verify C1 text morphology stability and run a one-seed text morphology loss-weight pilot sweep.
- Do not introduce a new architecture module.
- Do not modify labels, splits, manifests, image paths, report text, bio values, or task definition.
- Do not modify `dmea_ht/models.py`, `dmea_ht/data.py`, or `train.py`.
- Do not enable image morphology BCE, text negative loss, bio evidence loss, discordance-state loss, counterfactual loss, matched SupCon, or new anchor-fusion losses.
- Keep validation AUC as the primary decision metric.
- Treat test metrics as reporting-only.
- Reuse Phase C3 decision-gate logic and selected structural shortcut residual auditing.

### Planned Changes

- Add a C1 text morphology extended-seed config using seeds `[0, 1, 2, 3, 4, 42, 3407]`.
- Add one-seed pilot configs for text morphology loss weights:
  - 0.005;
  - 0.01;
  - 0.03;
  - 0.05;
  - 0.07;
  - 0.10.
- Add `scripts/collect_phase_c4_stability_and_pilots.py` to generate:
  - `analysis_reports/phase_c4/c1_extended_seed_summary.csv`;
  - `analysis_reports/phase_c4/c1_extended_seed_report.md`;
  - `analysis_reports/phase_c4/c1_weight_pilot_summary.csv`;
  - `analysis_reports/phase_c4/c1_weight_pilot_report.md`;
  - `analysis_reports/phase_c4/decision_gate_phase_c4_summary.csv`;
  - `analysis_reports/phase_c4/decision_gate_phase_c4_report.md`;
  - `analysis_reports/phase_c4/phase_c4_final_report.md`.

### Planned Validation

- Local static compile for the new C4 collector script.
- Server static compile in `/home/linruixin/chen/conda/envs/ma`.
- Server-side execution under `/home/linruixin/chen/project/DMEA-HT`.
- Training runs will use `/data/csb/DMEA-HT/HT_2025.12_25/manifest_distmatch_structmatch_evidence_v2.jsonl`.

### Actual Changes

- Added C4 extended-seed C1 config:
  - `configs/dmea_ht_v2_c4_c1_text_morphology_extended_seeds.yaml`.
- Added C4 one-seed pilot configs:
  - `configs/dmea_ht_v2_c4_c1_weight_w0005_pilot.yaml`;
  - `configs/dmea_ht_v2_c4_c1_weight_w001_pilot.yaml`;
  - `configs/dmea_ht_v2_c4_c1_weight_w003_pilot.yaml`;
  - `configs/dmea_ht_v2_c4_c1_weight_w005_pilot.yaml`;
  - `configs/dmea_ht_v2_c4_c1_weight_w007_pilot.yaml`;
  - `configs/dmea_ht_v2_c4_c1_weight_w010_pilot.yaml`.
- Added Phase C4 report collector:
  - `scripts/collect_phase_c4_stability_and_pilots.py`.
- No changes were made to:
  - `dmea_ht/models.py`;
  - `dmea_ht/data.py`;
  - `train.py`.

### Launch Status

- Local static compile passed:
  - `python -m py_compile scripts/collect_phase_c4_stability_and_pilots.py`.
- Local config parsing passed for all C4 configs.
- Code was pushed to GitHub at:
  - `d61f401`.
- Server was synced by bundle because GitHub TLS pull was unreliable in Phase C3.
- Server static compile passed.
- Server background driver launched:
  - `/home/linruixin/chen/project/DMEA-HT/phase_c4_driver_20260706.sh`.
- Server background PID:
  - initial launch `2012593` failed immediately because the generated shell driver expanded `$PY`, `$MANIFEST`, and `$OUT_DATE` while writing the script.
  - fixed driver relaunched as PID `2157540`.
- Server driver log:
  - `/home/linruixin/chen/project/DMEA-HT/analysis_reports/phase_c4/phase_c4_driver_20260706.log`.
  - failed initial launch log preserved as `/home/linruixin/chen/project/DMEA-HT/analysis_reports/phase_c4/phase_c4_driver_20260706.failed.log`.
- Planned run directories:
  - `/home/linruixin/chen/project/DMEA-HT/runs/dmea_ht_v2_c4_c1_text_morphology_extended_seeds_20260706`;
  - `/home/linruixin/chen/project/DMEA-HT/runs/dmea_ht_v2_c4_c1_weight_w0005_pilot_20260706`;
  - `/home/linruixin/chen/project/DMEA-HT/runs/dmea_ht_v2_c4_c1_weight_w001_pilot_20260706`;
  - `/home/linruixin/chen/project/DMEA-HT/runs/dmea_ht_v2_c4_c1_weight_w003_pilot_20260706`;
  - `/home/linruixin/chen/project/DMEA-HT/runs/dmea_ht_v2_c4_c1_weight_w005_pilot_20260706`;
  - `/home/linruixin/chen/project/DMEA-HT/runs/dmea_ht_v2_c4_c1_weight_w007_pilot_20260706`;
  - `/home/linruixin/chen/project/DMEA-HT/runs/dmea_ht_v2_c4_c1_weight_w010_pilot_20260706`.

### Remaining Issues

- Phase C4 training/report generation is running in the background.
- No final C4 metrics are available yet.
- Do not use test metrics for any Phase C4 selection decision once results are available.
