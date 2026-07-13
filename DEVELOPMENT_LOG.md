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

### Completed Results (2026-07-07)

- Server Phase C4 driver completed and generated all expected reports under:
  - `/home/linruixin/chen/project/DMEA-HT/analysis_reports/phase_c4`.
- Generated report files:
  - `c1_extended_seed_summary.csv`;
  - `c1_extended_seed_report.md`;
  - `c1_weight_pilot_summary.csv`;
  - `c1_weight_pilot_report.md`;
  - `decision_gate_phase_c4_summary.csv`;
  - `decision_gate_phase_c4_report.md`;
  - `phase_c4_final_report.md`.

### Extended-Seed Stability Summary

- Extended seeds:
  - `[0, 1, 2, 3, 4, 42, 3407]`.
- Validation AUC by seed:
  - seed 0: 0.8040;
  - seed 1: 0.7515;
  - seed 2: 0.7868;
  - seed 3: 0.7397;
  - seed 4: 0.7895;
  - seed 42: 0.7379;
  - seed 3407: 0.7931.
- Extended-seed validation AUC:
  - mean: 0.7718;
  - std: 0.0278;
  - median: 0.7868;
  - min/max: 0.7379 / 0.8040.
- Extended-seed validation AUPRC:
  - mean: 0.7726;
  - std: 0.0371.
- Maximum validation prediction/shortcut residual Spearman:
  - 0.2744.
- Phase C4 stability decision:
  - `STABILITY_FAIL`.
- Failure reason:
  - multiple seeds fell below the strict MVP reference validation AUC.

### Pilot Weight Sweep Summary

- Pilot seed:
  - seed 0.
- Pilot validation results:
  - weight 0.005: val AUC 0.8049, val AUPRC 0.8153, residual 0.2213, `PILOT_PASS_RECOMMEND_FORMAL`;
  - weight 0.010: val AUC 0.8049, val AUPRC 0.8156, residual 0.2248, `PILOT_PASS_RECOMMEND_FORMAL`;
  - weight 0.030: val AUC 0.8026, val AUPRC 0.8145, residual 0.2276, `PILOT_FAIL`;
  - weight 0.050: val AUC 0.8008, val AUPRC 0.8128, residual 0.2361, `PILOT_FAIL`;
  - weight 0.070: val AUC 0.7990, val AUPRC 0.8108, residual 0.2312, `PILOT_FAIL`;
  - weight 0.100: val AUC 0.7981, val AUPRC 0.8101, residual 0.2406, `PILOT_FAIL`.

### Final C4 Decision

- Do not claim a new stable improvement from Phase C4.
- C1 text morphology only should be marked unstable under the extended internal seed check.
- The current C1 status should be treated as:
  - not safely promotable as a stable main candidate without further analysis.
- Although weights 0.005 and 0.010 passed the seed-0 pilot gate, they should not automatically move to formal three-seed evaluation while the base C1 stability gate is failing.
- Next phase should be analysis-first:
  - inspect seed-level failure modes for seeds 1, 3, and 42;
  - compare their predictions against strict MVP and original C1;
  - audit whether the instability comes from optimization variance, split-specific sensitivity/specificity imbalance, evidence-label dependence, or shortcut residual coupling.
- Test metrics remain reporting-only and were not used for the Phase C4 decision.

## 2026-07-07 DMEA-v2 Phase C5 C1 Seed Failure Mode Diagnosis

### Plan

- Keep Phase C5 analysis-only.
- Do not introduce model modules or launch training.
- Do not modify `dmea_ht/models.py`, `dmea_ht/data.py`, `train.py`, manifests, labels, splits, image paths, report text, or bio values.
- Diagnose why C1 text morphology only fails extended-seed stability, especially seeds 1, 3, and 42.
- Use validation split for all diagnostic decisions.
- Treat test metrics as reporting-only if referenced at all.
- Compare:
  - strict MVP run;
  - original C1 text morphology only run;
  - C4 extended-seed C1 run.

### Planned Changes

- Add `scripts/analyze_c1_seed_failure_modes.py`.
- Add `scripts/analyze_c1_loss_dynamics.py`.
- Add `scripts/collect_phase_c5_report.py`.
- Generate Phase C5 reports under:
  - `analysis_reports/phase_c5/`.

### Planned Validation

- Local static compile:
  - `python -m py_compile scripts/analyze_c1_seed_failure_modes.py scripts/analyze_c1_loss_dynamics.py scripts/collect_phase_c5_report.py`.
- Server static compile in `/home/linruixin/chen/conda/envs/ma`.
- Server-side analysis execution using completed run outputs only.

### Actual Changes

- Added `scripts/analyze_c1_seed_failure_modes.py`.
- Added `scripts/analyze_c1_loss_dynamics.py`.
- Added `scripts/collect_phase_c5_report.py`.
- No changes were made to:
  - `dmea_ht/models.py`;
  - `dmea_ht/data.py`;
  - `train.py`;
  - manifests;
  - labels or splits.

### Validation Results

- Local static compile passed.
- Server static compile passed in `/home/linruixin/chen/conda/envs/ma`.
- Server analysis completed under:
  - `/home/linruixin/chen/project/DMEA-HT/analysis_reports/phase_c5`.
- Initial server run exposed two environment/data-format issues, both fixed without adding dependencies:
  - removed pandas `to_markdown()` dependency on optional `tabulate`;
  - handled list-valued `matched_morphology_terms`.

### Generated Reports

- `c1_seed_failure_summary.csv`
- `c1_seed_failure_report.md`
- `c1_vs_mvp_patient_delta_val.csv`
- `c1_vs_mvp_patient_delta_report.md`
- `c1_vs_mvp_stratified_delta.csv`
- `c1_vs_mvp_stratified_delta_report.md`
- `c1_prediction_distribution_by_seed.csv`
- `c1_prediction_distribution_report.md`
- `c1_loss_dynamics_by_seed.csv`
- `c1_loss_dynamics_report.md`
- `c1_seed_shortcut_residual.csv`
- `c1_seed_shortcut_residual_report.md`
- `phase_c5_final_report.md`

### Key Findings

- Good seed mean validation AUC:
  - 0.7933.
- Bad seed mean validation AUC:
  - 0.7430.
- Good seed mean positive-negative prediction gap:
  - 0.2168.
- Bad seed mean positive-negative prediction gap:
  - 0.1430.
- Bad seed mean validation sensitivity/specificity:
  - 0.5603 / 0.7660.
- Good seed mean validation sensitivity/specificity:
  - 0.5851 / 0.8085.
- Bad seed maximum residual shortcut Spearman:
  - 0.2291.
- Good seed maximum residual shortcut Spearman:
  - 0.2744.
- Therefore, bad seeds do not appear to fail because of stronger selected-structure shortcut residual coupling.
- C1 vs MVP validation patient-delta analysis:
  - for bad seeds and negative labels, mean abs-error delta was -0.1716, so C1 helps negatives by pushing probabilities downward;
  - for bad seeds and positive labels, mean abs-error delta was +0.2083, so C1 strongly harms positives;
  - for good seeds and positive labels, mean abs-error delta was also positive but smaller at +0.1427.
- Loss/checkpoint diagnostics:
  - bad seeds selected earlier best epochs: seed 1 epoch 14, seed 3 epoch 10, seed 42 epoch 13;
  - good seeds generally selected later epochs: seed 0 epoch 30, seed 2 epoch 28, seed 4 epoch 28, seed 3407 epoch 23;
  - bad seeds had higher available `text_morphology_loss` in final best-state metrics.
- Per-epoch train/validation loss curves are not currently logged, so overfitting timing and evidence-loss dominance cannot be proven from existing artifacts.

### Final C5 Decision

- C1 text morphology only remains unstable and should not be treated as a stable main model.
- Most likely failure cause:
  - optimization variance / checkpoint instability, with C1 tending to push probabilities downward and disproportionately harm positive cases in bad seeds.
- Less likely primary cause:
  - residual selected-structure shortcut coupling.
- Recommended next phase direction:
  - `A. Optimization stabilization first`.
- Do not start a new architecture phase.
- If a follow-up training phase is approved, first add better training-curve logging and run a small stabilization pilot rather than expanding evidence losses.

## 2026-07-07 DMEA-v2 Phase C6 Optimization Stabilization and Positive Preservation Pilot

### Plan

- Keep Phase C6 focused on stabilizing existing C1 text morphology supervision.
- Do not add architecture modules.
- Do not modify labels, splits, manifests, image paths, report text, bio values, or shortcut handling.
- Modify `train.py` only for:
  - per-epoch diagnostics;
  - `text_morphology_start_epoch`;
  - positive/negative validation probability summaries.
- Add bad-seed pilot configs for seeds `[1, 3, 42]`:
  - text morphology weight 0.005;
  - text morphology weight 0.010;
  - delayed text morphology start at epoch 5 with weight 0.010.
- Add a C6 collector to compare bad-seed pilots against:
  - strict MVP reference;
  - original C1 bad-seed performance;
  - C4 extended C1 bad-seed subset.
- Use validation metrics only for decisions.
- Treat test metrics as reporting-only.

### Planned Validation

- Local static compile:
  - `python -m py_compile scripts/collect_phase_c6_stabilization_report.py train.py`.
- Server static compile under `/home/linruixin/chen/conda/envs/ma`.
- Server-side pilot execution using:
  - `/data/csb/DMEA-HT/HT_2025.12_25/manifest_distmatch_structmatch_evidence_v2.jsonl`.

### Actual Changes

- Updated `train.py` with per-epoch diagnostics:
  - `metrics_by_epoch.csv`;
  - train/validation total loss;
  - classification loss;
  - text morphology loss;
  - validation AUC/AUPRC and threshold metrics;
  - validation positive/negative probability means;
  - validation positive-negative prediction gap;
  - selected-by-validation-AUC epoch flag.
- Added config-controlled delayed text morphology loss:
  - `loss.text_morphology_start_epoch`;
  - default-compatible behavior is epoch 0.
- Added C6 bad-seed pilot configs:
  - `configs/dmea_ht_v2_c6_badseed_weight_w0005.yaml`;
  - `configs/dmea_ht_v2_c6_badseed_weight_w001.yaml`;
  - `configs/dmea_ht_v2_c6_badseed_delay_text_morphology.yaml`.
- Added C6 collector:
  - `scripts/collect_phase_c6_stabilization_report.py`.

### Launch Status

- Local static compile passed.
- Local C6 config parsing passed.
- Code was pushed to GitHub at:
  - `2b44c36`.
- Server synced by bundle.
- Server static compile passed.
- Server background driver launched:
  - `/home/linruixin/chen/project/DMEA-HT/phase_c6_driver_20260707.sh`.
- Server background PID:
  - `2956962`.
- Server driver log:
  - `/home/linruixin/chen/project/DMEA-HT/analysis_reports/phase_c6/phase_c6_driver_20260707.log`.
- Planned C6 run directories:
  - `/home/linruixin/chen/project/DMEA-HT/runs/dmea_ht_v2_c6_badseed_weight_w0005_20260707`;
  - `/home/linruixin/chen/project/DMEA-HT/runs/dmea_ht_v2_c6_badseed_weight_w001_20260707`;
  - `/home/linruixin/chen/project/DMEA-HT/runs/dmea_ht_v2_c6_badseed_delay_text_morphology_20260707`.

### Remaining Issues

- C6 pilots finished on the server.
- Reports were generated under:
  - `/home/linruixin/chen/project/DMEA-HT/analysis_reports/phase_c6`.
- Test metrics remain reporting-only and were not used for C6 decisions.

### Final Results

- A collector ranking bug was found after the first report pass:
  - string sorting placed `STABILIZATION_FAIL` before `STABILIZATION_PARTIAL_NEEDS_MORE_ANALYSIS`;
  - the collector was fixed to rank decisions explicitly as PASS, PARTIAL, then FAIL;
  - reports were regenerated on the server after the fix.
- C6 validation-only summary:
  - `delay_w001_start5`:
    - validation AUC 0.7450 +/- 0.0070;
    - validation AUPRC 0.7254;
    - sensitivity/specificity 0.6028 / 0.7234;
    - positive-negative prediction gap 0.1548;
    - max absolute shortcut residual Spearman 0.2257;
    - selected epoch mean/min/max 16.3 / 13 / 22;
    - decision `STABILIZATION_PARTIAL_NEEDS_MORE_ANALYSIS`.
  - `w001`:
    - validation AUC 0.7438 +/- 0.0090;
    - validation AUPRC 0.7282;
    - sensitivity/specificity 0.5816 / 0.7518;
    - positive-negative prediction gap 0.1576;
    - max absolute shortcut residual Spearman 0.2225;
    - selected epoch mean/min/max 16.3 / 13 / 22;
    - decision `STABILIZATION_PARTIAL_NEEDS_MORE_ANALYSIS`.
  - `w0005`:
    - validation AUC 0.7421 +/- 0.0135;
    - validation AUPRC 0.7288;
    - sensitivity/specificity 0.6738 / 0.6525;
    - positive-negative prediction gap 0.1531;
    - max absolute shortcut residual Spearman 0.1983;
    - selected epoch mean/min/max 13.7 / 10 / 17;
    - decision `STABILIZATION_FAIL`.
- Positive preservation audit vs strict MVP:
  - all candidates still helped many negative-label patients by lowering predicted probabilities;
  - all candidates still harmed most positive-label patients by lowering positive probabilities relative to MVP;
  - `w0005` harmed positives least but failed the validation AUC gate;
  - `delay_w001_start5` had the best C6 validation AUC but still failed to reach the strict MVP reference.
- Shortcut residual audit:
  - no candidate showed an alarming prediction-shortcut residual rank correlation;
  - the main remaining problem is positive preservation / optimization behavior, not obvious residual shortcut dominance.

### Final C6 Decision

- No C6 candidate reached `STABILIZATION_PASS_RECOMMEND_FORMAL`.
- The best ranked candidate is `delay_w001_start5`, but it is only `STABILIZATION_PARTIAL_NEEDS_MORE_ANALYSIS`.
- C6 does not justify launching a formal evaluation run.
- C1 text morphology supervision should remain unstable / ablation-only for now.
- Recommended next step:
  - do not expand architecture;
  - either demote C1 from the main path, or run a very small follow-up focused specifically on positive preservation / calibration rather than stronger evidence loss.

## 2026-07-07 DMEA-v2 Phase C7 Demote Unstable Evidence BCE and Re-center Main Path

### Plan

- Keep Phase C7 analysis/documentation-only.
- Do not train.
- Do not modify model/data/training code.
- Consolidate C1-C6 evidence and formally demote unstable text morphology BCE branches.
- Restore strict structural matched DMEA-MVP as the current main path.
- Update future decision gate with positive-preservation requirements.

### Actual Changes

- Added `scripts/collect_phase_c7_route_correction_report.py`.
- Generated Phase C7 reports under `analysis_reports/phase_c7`.
- No model/data/training changes.
- No new training launched.

### Generated Reports

- `main_path_decision_summary.csv`
- `evidence_bce_failure_timeline.csv`
- `evidence_bce_failure_report.md`
- `ablation_status_table.csv`
- `positive_preservation_summary.csv`
- `decision_gate_update.md`
- `phase_c7_final_report.md`
- `inputs_used_and_missing.csv`
- `phase_c6_csv_appendix.csv`

### Decision

- Current main path: strict structural matched DMEA-MVP.
- C1 text morphology BCE: ablation-only / unstable.
- C1 text + image evidence: failed ablation.
- C2 text anchor: failed ablation.
- C6 stabilization candidates: no formal evaluation justified.
- Future weak-evidence-supervised candidates must pass a positive-preservation gate and bad-seed pilot before formal training.

## 2026-07-07 DMEA-v2 Phase C8 Strict MVP Evidence Diagnostics and Error Taxonomy

### Plan

- Keep Phase C8 analysis-only.
- Do not train or modify model/data/training code.
- Analyze strict structural matched DMEA-MVP as the current main path.
- Generate patient-level error taxonomy, evidence strata, high-confidence error review, and shortcut-audit strata.
- Use validation split for all route decisions.
- Treat test metrics as reporting-only.

### Actual Changes

- Added `scripts/analyze_strict_mvp_error_taxonomy.py`.
- Added `scripts/analyze_strict_mvp_evidence_strata.py`.
- Added `scripts/collect_phase_c8_report.py`.
- No model/data/training changes.
- No new training launched.

### Validation Results

- Local static compile passed.
- Server static compile passed under `/home/linruixin/chen/conda/envs/ma`.
- Server C8 analysis completed under:
  - `/home/linruixin/chen/project/DMEA-HT/analysis_reports/phase_c8`.

### Generated Reports

- `strict_mvp_error_cases_val.csv`
- `strict_mvp_error_cases_test_reporting_only.csv`
- `strict_mvp_error_taxonomy_summary.csv`
- `strict_mvp_error_taxonomy_report.md`
- `strict_mvp_evidence_strata_val.csv`
- `strict_mvp_evidence_strata_test_reporting_only.csv`
- `strict_mvp_high_confidence_errors_val.csv`
- `strict_mvp_high_confidence_errors_test_reporting_only.csv`
- `strict_mvp_shortcut_strata_val.csv`
- `strict_mvp_evidence_diagnostics_report.md`
- `strict_mvp_overall_metrics.csv`
- `phase_c8_final_report.md`

### Key Findings

- Validation-only strict MVP recap:
  - AUC 0.7443;
  - AUPRC 0.7192;
  - sensitivity/specificity at threshold 0.5: 0.8865 / 0.4326;
  - positive-negative prediction gap: 0.1797;
  - validation false negatives / false positives: 16 / 80.
- Validation error taxonomy:
  - `long_report_or_multivisit_uncertainty`: 25 errors, all false positives;
  - `other_error`: 25 errors, 1 false negative and 24 false positives;
  - `morphology_positive_false_negative`: 15 errors, all false negatives;
  - `high_confidence_false_positive`: 12 errors;
  - `morphology_low_confidence_false_positive`: 10 errors;
  - `borderline_error`: 9 errors.
- Evidence strata:
  - morphology-positive stratum had AUC 0.7331 and high false-positive rate 0.5983;
  - negative-evidence-positive stratum had specificity 0.2745 and false-positive rate 0.7255;
  - medium negative-confidence stratum had weaker ranking, AUC 0.5767 and gap 0.0518.
- High-confidence errors:
  - 13 validation high-confidence error rows were found;
  - these were dominated by high-confidence false positives.
- Selected structural audit observations:
  - no validation audit bin exceeded the configured large-concentration threshold;
  - report-length top quartile had weaker AUC 0.6094 and higher error rate 0.4348, but this is a diagnostic caution signal only;
  - selected visit/image high bins had lower AUC 0.6809 than low bins 0.8581, also audit-only and not causal evidence.

### Final C8 Decision

- Current main path remains: strict structural matched DMEA-MVP.
- C1/C2/C6 remain ablation-only.
- Next-phase recommendation: `RETURN_TO_DATA_AUDIT`.
- Test metrics remain reporting-only and were not used for the Phase C8 decision.

## 2026-07-07 DMEA-v2 Phase C9 Strict MVP False-Positive Data Audit

### Plan

- Keep Phase C9 analysis-only.
- Do not train or modify model/data/training code.
- Use strict structural matched DMEA-MVP as the current main path.
- Focus on validation false positives identified by Phase C8:
  - long-report false positives;
  - multi-visit false positives;
  - negative-evidence-positive false positives;
  - high-confidence false positives;
  - morphology/negative evidence overlap;
  - possible patient-level report aggregation artifacts.
- Treat shortcut fields as audit-only.
- Use test outputs only as reporting-only if inspected.
- Produce a data-audit report before proposing any model or data-construction change.

### Actual Changes

- Added `scripts/analyze_phase_c9_false_positive_data_audit.py`.
- Generated Phase C9 reports under `analysis_reports/phase_c9`.
- No model/data/training changes.
- No new training launched.

### Generated Reports

- `c9_fp_patient_audit_val.csv`
- `c9_fp_flag_summary_val.csv`
- `c9_fp_high_priority_cases_val.csv`
- `phase_c9_final_report.md`

### Key Findings

- Unique validation false-positive patients:
  - 37.
- False-positive patients present across all three formal seeds:
  - 20 / 37.
- False-positive patients with at least one high-confidence false-positive seed:
  - 9 / 37.
- False-positive patients with morphology/negative-evidence overlap:
  - 19 / 37.
- False-positive patients with `txt_negative_label=1`:
  - 14 / 37.
- False-positive patients with suspected aggregation artifact:
  - 16 / 37.
- High-priority examples show repeated cases where positive morphology terms such as low/uneven echo coexist with negative terms such as uniform echo or no abnormal blood flow across concatenated longitudinal reports.

### C9 Decision

- Recommendation: `DATA_CONSTRUCTION_AUDIT_BEFORE_MODEL_CHANGE`.
- Do not start a model change yet.
- Before any training pilot, manually review high-priority false-positive patients and verify whether positive morphology terms are historical, negated, contradicted, benign nodular findings, or mixed with later negative evidence.
- Strict MVP remains the current main path.
- C1/C2/C6 remain ablation-only.

## 2026-07-07 DMEA-v2 Phase C10 False-Positive Report Source Audit

### Plan

- Keep Phase C10 analysis-only.
- Do not train or modify model/data/training code.
- Use C9 high-priority strict MVP false-positive patients as the audit target.
- Split patient-level concatenated reports into visit-level blocks where possible.
- Audit whether false-positive evidence is driven by:
  - thyroid-relevant morphology text;
  - non-thyroid sections such as breast, carotid, cardiac, abdominal, pelvic, or urinary reports;
  - benign/nodular thyroid findings rather than diffuse HT-like thyroiditis;
  - historical positive text contradicted by later negative or benign text;
  - morphology/negative evidence overlap inside the same patient history.
- Treat all structural fields as audit-only.
- Use validation errors only for route decisions.

### Actual Changes

- Added `scripts/analyze_phase_c10_fp_report_source_audit.py`.
- Generated Phase C10 reports under `analysis_reports/phase_c10`.
- No model/data/training changes.
- No new training launched.

### Generated Reports

- `c10_fp_visit_source_audit_val.csv`
- `c10_fp_patient_source_summary_val.csv`
- `c10_fp_source_flag_summary_val.csv`
- `phase_c10_final_report.md`

### Key Findings

- False-positive patients audited:
  - 37.
- Patients with any thyroid morphology hit:
  - 27 / 37.
- Patients with non-thyroid morphology hit:
  - 6 / 37.
- Patients with thyroid positive/negative overlap:
  - 17 / 37.
- Patients with benign/nodular mimic signal:
  - 8 / 37.
- Patients whose latest visit has negative thyroid cues:
  - 24 / 37.
- Patients with early positive and latest negative conflict:
  - 12 / 37.
- Highest-priority examples include repeated false positives where low/uneven echo or nodule language coexists with negative or benign thyroid cues across longitudinal visits.

### C10 Decision

- Recommendation: `AUDIT_TEMPORAL_EVIDENCE_CONFLICT_BEFORE_TRAINING`.
- Do not start training yet.
- Before any pilot, define and audit a report-construction or evidence-filtering hypothesis that targets temporal evidence conflict without changing labels, splits, or feeding shortcut variables into the classifier.
- Strict MVP remains the current main path.
- C1/C2/C6 remain ablation-only.

## 2026-07-07 DMEA-v2 Phase C11 Report-Filter Hypothesis Audit

### Plan

- Keep Phase C11 analysis-only.
- Do not train or modify model/data/training code.
- Define candidate report-construction hypotheses from C9/C10 findings:
  - latest negative thyroid evidence suppresses historical positive morphology;
  - benign/nodule morphology without latest diffuse HT-like signal;
  - require latest diffuse HT-like signal for morphology-positive evidence;
  - non-thyroid-only morphology source;
  - thyroid positive/negative overlap review.
- Audit each hypothesis on the validation cohort, not just false positives.
- For every hypothesis, report false-positive capture and label-positive patient flag rate as a positive-preservation risk proxy.
- Only consider a later pilot if a hypothesis captures false positives without flagging many label-positive patients.
- Test metrics remain unused.

### Actual Changes

- Added `scripts/analyze_phase_c11_report_filter_hypotheses.py`.
- Generated Phase C11 reports under `analysis_reports/phase_c11`.
- No model/data/training changes.
- No new training launched.

### Generated Reports

- `c11_report_filter_patient_table_val.csv`
- `c11_report_filter_hypothesis_summary_val.csv`
- `c11_positive_preservation_risk_val.csv`
- `phase_c11_final_report.md`

### Key Findings

- Validation patients audited:
  - 94.
- Mean-threshold false-positive patients:
  - 24.
- Label-positive patients used for positive-preservation risk:
  - 47.
- `benign_nodule_without_latest_diffuse`:
  - captured 11 / 24 mean-threshold false positives;
  - flagged 5 / 47 label-positive patients;
  - recommendation status: `PILOT_ELIGIBLE_LOW_POSITIVE_RISK`.
- `require_latest_diffuse_ht_like`:
  - captured 9 / 24 mean-threshold false positives;
  - flagged 4 / 47 label-positive patients;
  - recommendation status: `PILOT_ELIGIBLE_LOW_POSITIVE_RISK`.
- `positive_negative_overlap_review` captured false positives but also flagged 35 / 47 label-positive patients, so it remains audit-only.
- `latest_negative_suppresses_history` also has high positive-preservation risk and should not be used as a broad filter yet.
- `non_thyroid_morphology_only` flagged no label-positive patients but only captured 3 / 24 false positives, so it is case-review only.

### C11 Decision

- Recommendation: `ALLOW_REPORT_FILTER_PILOT_FOR_LOW_RISK_HYPOTHESIS`.
- Next step may be a low-cost report-construction pilot using only low-risk hypotheses, especially:
  - benign/nodule morphology without latest diffuse HT-like evidence;
  - requiring latest diffuse HT-like evidence before treating morphology as HT-positive evidence.
- Any pilot must keep patient-level split, task definition, labels, and test isolation unchanged.
- Shortcut/audit variables must remain outside the classifier.
- A pilot must be validation-selected, stress-seed checked, and followed by positive-preservation plus shortcut residual audits.

## 2026-07-07 DMEA-v2 Phase C12 Report-Construction Pilot

### Plan

- Keep C12 as a low-cost report-construction pilot before any architecture change.
- Build a new manifest from the strict structural matched evidence manifest.
- Do not change patient IDs, labels, split assignment, image paths, bio values, or task definition.
- Do not use test predictions or labels for selecting the pilot.
- Do not use model predictions inside the report filter.
- Apply only deterministic text-construction rules motivated by C11 low-risk hypotheses:
  - remove benign/nodule thyroid morphology clauses when the latest thyroid visit lacks diffuse HT-like evidence;
  - optionally require latest diffuse HT-like evidence before retaining thyroid morphology clauses as HT-positive report evidence.
- Preserve negative thyroid clauses and diagnostic/diffuse HT-like clauses.
- Recompute evidence weak labels after filtering because text labels must match the pilot report text.
- Write a manifest-level audit report covering:
  - row/split/label invariance;
  - report length changes by split and label;
  - text evidence label changes by split and label;
  - positive-preservation risk using validation patients.
- Only launch training after the C12 manifest audit shows the pilot did not change labels/splits and has a defensible positive-preservation profile.

### Actual Changes

- Added `scripts/build_phase_c12_report_filter_pilot_manifest.py`.
- Added `configs/dmea_ht_v2_c12_report_filter_pilot.yaml`.
- Built server manifest:
  - `/data/csb/DMEA-HT/HT_2025.12_25/manifest_distmatch_structmatch_evidence_v2_c12_report_filter_pilot.jsonl`.
- Generated manifest audit reports under `analysis_reports/phase_c12`.
- Launched C12 single-seed training pilot on server:
  - PID: `1370618`;
  - log: `phase_c12_train_20260707_153025.log`;
  - output dir: `runs/dmea_ht_v2_c12_report_filter_pilot`.

### Generated Reports

- `c12_report_filter_patient_audit.csv`
- `c12_report_filter_split_label_summary.csv`
- `c12_report_filter_label_change_summary.csv`
- `c12_report_filter_positive_preservation_val.csv`
- `phase_c12_manifest_audit_report.md`

### Key Findings

- Input rows:
  - 780.
- Output rows:
  - 780.
- Invariance issues:
  - 0.
- Split/label counts remained unchanged:
  - train: 301 / 301;
  - val: 47 / 47;
  - test: 42 / 42.
- Validation label-positive preservation risk:
  - filtered positives: 0 / 47;
  - `txt_morphology_label` changed positives: 0 / 47;
  - `image_morphology_weak_label` changed positives: 0 / 47.
- Validation label-negative impact:
  - filtered negatives: 4 / 47;
  - `txt_morphology_label` changed negatives: 2 / 47.
- Recommendation:
  - `ALLOW_C12_SINGLE_SEED_TRAINING_PILOT`.

### C12 Decision

- Manifest construction passed the invariance and validation positive-preservation gates.
- A single-seed training pilot has been launched, but no AUC conclusion is available yet.
- Do not select a final model or start stress seeds until C12 single-seed validation metrics are collected and audited.
- Test metrics remain reporting-only and must not drive pilot selection.

### C12 Single-Seed Training Result

- C12 single-seed pilot completed on server.
- Seed:
  - 0.
- Best epoch:
  - 25.
- Validation AUC / AUPRC:
  - 0.7936 / 0.8055.
- Validation sensitivity / specificity at threshold 0.5:
  - 0.5745 / 0.7872.
- Validation FN / FP:
  - 20 / 10.
- Validation positive-negative probability gap:
  - 0.2047.
- Main validation error type:
  - `morphology_positive_false_negative`: 18 / 30 validation errors.
- Shortcut residual audit:
  - validation max abs Spearman: 0.2394;
  - validation linear R2 from shortcut fields: 0.0821;
  - validation shortcut-only label AUC audit-only: 0.3332.

### C12 Stress-Seed Decision

- Recommendation: `ALLOW_C12_STRESS_SEED_PILOT`.
- Rationale:
  - single-seed validation AUC improved over the strict MVP C8 route;
  - false positives are reduced, consistent with the report-filter hypothesis;
  - shortcut-only label AUC remains below chance in the audit;
  - false negatives now dominate, so the pilot cannot be promoted from one seed.
- Added `configs/dmea_ht_v2_c12_report_filter_stress_seeds.yaml` for seeds `[1, 3, 42]`.
- Stress-seed results must be collected before any formal selection or model claim.

## 2026-07-07 DMEA-v2 Phase C13 FN Recall Audit

### Plan

- Keep Phase C13 analysis-only while C12 stress seeds are running.
- Use C12 single-seed validation errors to diagnose the dominant false-negative mode.
- Do not train, modify labels, modify split assignment, or use test for selection.
- Audit whether the C12 false negatives are explained by:
  - C12 report filtering damaging validation-positive evidence;
  - high-confidence morphology text not being translated into positive predictions;
  - negative evidence suppressing positives;
  - long-report or multi-visit aggregation;
  - bio missingness.
- Use this audit only to decide the next pilot design after stress-seed results are collected.

### Actual Changes

- Added `scripts/analyze_phase_c13_fn_recall_audit.py`.
- Generated Phase C13 reports under `analysis_reports/phase_c13_fn_recall_audit`.
- No model/data/training changes.
- C12 stress seeds remain running separately.

### Generated Reports

- `c13_error_type_summary.csv`
- `c13_fn_feature_summary_val.csv`
- `c13_lowest_probability_fn_cases_val.csv`
- `phase_c13_fn_recall_audit_report.md`

### Key Findings

- C12 validation errors:
  - 30.
- C12 validation FN / FP:
  - 20 / 10.
- Main validation error type:
  - `morphology_positive_false_negative`: 18 / 30 errors.
- C12 filter positive-damage check:
  - validation label-positive filtered patients: 0 / 47;
  - validation label-positive `txt_morphology_label` changes: 0 / 47.
- FN concentration:
  - high morphology confidence: 13 FN;
  - report length q4 high: 8 FN, false-negative rate 0.6667;
  - selected visit high bin: 18 FN, false-negative rate 0.5806.
- Negative evidence is not a sufficient global explanation:
  - `txt_negative_label=0` contains 17 FN;
  - `txt_negative_label=1` contains 3 FN.

### C13 Decision

- Recommendation: `DESIGN_C13_TEMPORAL_OR_LONG_REPORT_RECALL_PILOT_AFTER_STRESS_SEEDS`.
- Do not start a C13 training pilot until C12 stress-seed metrics are collected.
- If stress seeds confirm the same FN-heavy pattern, the next pilot should target temporal or long-report recall rather than undoing the C12 false-positive report filter.

## 2026-07-07 DMEA-v2 Phase C12 Stress-Seed Result

### Actual Changes

- Collected C12 stress-seed training outputs from `runs/dmea_ht_v2_c12_report_filter_stress_seeds`.
- Generated stress-seed reports under `analysis_reports/phase_c12_stress`.
- No model or data construction changes were made in this collection step.

### Generated Reports

- `c12_stress_metrics_by_seed.csv`
- `c12_stress_metrics_summary.csv`
- `c12_stress_confusion_matrix_by_seed.csv`
- `strict_mvp_error_taxonomy_summary.csv`
- `strict_mvp_evidence_strata_val.csv`
- `shortcut_residual/shortcut_residual_audit.csv`
- `phase_c12_stress_decision_report.md`

### Key Findings

- Validation AUC by stress seed:
  - seed 1: 0.7773;
  - seed 3: 0.7691;
  - seed 42: 0.7429.
- Validation AUC mean / std:
  - 0.7631 / 0.0180.
- Validation AUPRC mean / std:
  - 0.7794 / 0.0251.
- Seed 42 has high sensitivity but very low specificity:
  - sensitivity 0.9574;
  - specificity 0.1064;
  - FP 42.
- Stress error taxonomy:
  - `morphology_positive_false_negative`: 28 errors;
  - `long_report_or_multivisit_uncertainty`: 22 errors;
  - `high_confidence_false_positive`: 13 errors.
- Shortcut residual audit remains acceptable:
  - pooled validation max abs Spearman: 0.0946;
  - pooled validation linear R2 from shortcut fields: 0.0373;
  - pooled validation shortcut-only label AUC audit-only: 0.4918.

### Stress Decision

- Recommendation: `DO_NOT_PROMOTE_C12_FORMALLY`.
- C12 remains a useful report-construction direction but is not stable enough for formal model selection.
- Next action: `DESIGN_C13_TEMPORAL_FOCUS_REPORT_PILOT`.
- C13 should preserve the C12 false-positive filter while placing thyroid-relevant latest and diffuse/morphology clauses before full report text to reduce long-report and multi-visit truncation under `text_max_length=256`.

## 2026-07-07 DMEA-v2 Phase C13 Temporal-Focus Report Pilot

### Plan

- Build a new C13 manifest from the C12 report-filter manifest.
- Keep labels, split assignment, patient IDs, images, bio values, and task definition unchanged.
- Preserve the C12 false-positive report filter.
- Do not use labels, predictions, or test-selected information in the text construction rule.
- Add a deterministic report-text prefix containing thyroid-relevant latest and historical clauses before the full report text.
- Target the observed long-report and multi-visit failure mode under `text_max_length=256`.
- Recompute evidence weak labels after report text construction.
- Audit row/split/label invariance and whether thyroid morphology/diffuse evidence appears more often in the first 256 characters.
- Launch at most a single-seed pilot only if the manifest audit passes.

### Actual Changes

- Added `scripts/build_phase_c13_temporal_focus_manifest.py`.
- Added `configs/dmea_ht_v2_c13_temporal_focus_pilot.yaml`.
- Built server manifest:
  - `/data/csb/DMEA-HT/HT_2025.12_25/manifest_distmatch_structmatch_evidence_v2_c13_temporal_focus.jsonl`.
- Generated manifest audit reports under `analysis_reports/phase_c13_temporal_focus`.
- No model architecture, label, split, image, or bio changes.

### Generated Reports

- `c13_temporal_focus_patient_audit.csv`
- `c13_temporal_focus_split_label_summary.csv`
- `c13_temporal_focus_positive_focus_val.csv`
- `phase_c13_temporal_focus_manifest_audit_report.md`

### Key Findings

- Input rows:
  - 780.
- Output rows:
  - 780.
- Invariance issues:
  - 0.
- Split/label counts remained unchanged:
  - train: 301 / 301;
  - val: 47 / 47;
  - test: 42 / 42.
- Validation label-positive first-256 evidence exposure:
  - morphology mean before / after: 2.3191 / 2.8298;
  - diffuse mean before / after: 0.7447 / 1.5532.
- Validation label-positive weak-label damage:
  - `txt_morphology_label` changed: 0 / 47;
  - `image_morphology_weak_label` changed: 0 / 47.

### C13 Manifest Decision

- Recommendation: `ALLOW_C13_SINGLE_SEED_TEMPORAL_FOCUS_PILOT`.
- The pilot is allowed because it preserves split/label invariants, does not change weak labels, and directly targets long-report truncation by increasing thyroid evidence exposure inside the model's first 256 text tokens.
- C13 remains a pilot until validation metrics and shortcut residuals are collected.

## 2026-07-10 DMEA-v2 Phase C13 Temporal-Focus Stress-Seed Result

### Actual Changes

- Ran C13 temporal-focus stress seeds with user-requested seeds `[0, 42, 3407]`.
- Collected outputs from `runs/dmea_ht_v2_c13_temporal_focus_stress_seeds`.
- Generated stress-seed reports under `analysis_reports/phase_c13_stress`.
- No label, split, task-definition, model-architecture, image, or bio changes were made in this collection step.
- Test metrics remain reporting-only and were not used for model selection.

### Generated Reports

- `c13_stress_metrics_by_seed.csv`
- `c13_stress_metrics_summary.csv`
- `c13_stress_confusion_matrix_by_seed.csv`
- `strict_mvp_error_taxonomy_summary.csv`
- `strict_mvp_evidence_strata_val.csv`
- `strict_mvp_shortcut_strata_val.csv`
- `shortcut_residual/shortcut_residual_audit.csv`
- `phase_c13_stress_decision_report.md`

### Key Findings

- Validation AUC by seed:
  - seed 0: 0.8656;
  - seed 42: 0.8746;
  - seed 3407: 0.8592.
- Validation AUC mean / std:
  - 0.8665 / 0.0077.
- Validation AUPRC mean / std:
  - 0.8570 / 0.0049.
- Validation sensitivity mean / std:
  - 0.6525 / 0.1568.
- Validation specificity mean / std:
  - 0.8511 / 0.0426.
- Test reporting-only AUC mean / std:
  - 0.8460 / 0.0077.
- Validation error taxonomy remains false-negative dominated:
  - `morphology_positive_false_negative`: 44 errors;
  - `long_report_or_multivisit_uncertainty`: 7 errors;
  - `other_error`: 7 errors;
  - `borderline_error`: 6 errors;
  - `high_confidence_false_positive`: 4 errors.
- Shortcut residual audit remains acceptable:
  - pooled validation max abs Spearman: 0.1549;
  - pooled validation linear R2 from shortcut fields: 0.0601;
  - pooled validation shortcut-only label AUC audit-only: 0.4762.

### C13 Stress Decision

- Recommendation: `PROMOTE_C13_AS_CURRENT_STRICT_BEST_NOT_FINAL`.
- C13 is now the strongest strict structural-matched single-model route observed so far.
- C13 should not be treated as final because validation AUC remains below the 0.90 target and sensitivity is still seed-sensitive.
- Next action: design a C14 low-cost pilot focused on morphology-positive false-negative recall and high-report-length recall, while preserving labels, patient-level split, task definition, and shortcut exclusion from the classifier.

## 2026-07-10 DMEA-v2 Phase C14-A FN Token Exposure Audit

### Plan

- Do not train, tune thresholds, change labels, change splits, change task definition, or edit model/data/training core code.
- Audit C13 temporal-focus stress-seed validation positives only for route selection.
- Use C13 stress run `runs/dmea_ht_v2_c13_temporal_focus_stress_seeds`.
- Use C13 manifest `/data/csb/DMEA-HT/HT_2025.12_25/manifest_distmatch_structmatch_evidence_v2_c13_temporal_focus.jsonl`.
- Classify positive cases as TP, FN, stable TP, stable FN, and seed-sensitive across seeds `[0, 42, 3407]`.
- Determine whether remaining morphology-positive false negatives are primarily due to missing first-window evidence or evidence being visible but underused.
- Keep test split reporting-only.

### Actual Changes

- Added `scripts/analyze_phase_c14a_fn_token_exposure.py`.
- Added `scripts/collect_phase_c14a_report.py`.
- Generated C14-A audit reports under `analysis_reports/phase_c14a`.
- Used the project tokenizer contract:
  - character-level tokenizer;
  - `text_max_length=256`;
  - effective report text window is 254 characters plus special tokens.
- No training was launched.
- No model, data loader, label, split, image, bio, or threshold changes were made.

### Generated Reports

- `c14a_positive_patient_token_exposure_val.csv`
- `c14a_fn_vs_tp_summary_val.csv`
- `c14a_cross_seed_stable_fn_cases_val.csv`
- `c14a_seed_sensitive_positive_cases_val.csv`
- `c14a_evidence_exposure_strata_val.csv`
- `c14a_seed_overlap_summary_val.csv`
- `c14a_token_exposure_audit_report.md`
- `phase_c14a_final_report.md`
- `inputs_used_and_missing.csv`
- `c14a_positive_patient_token_exposure_test_reporting_only.csv`

### Key Findings

- Validation positive rows:
  - FN rows: 49 across 23 patients;
  - TP rows: 92 across 40 patients.
- Cross-seed patient categories:
  - stable FN patients: 19;
  - stable TP patients: 28.
- Stable FN evidence exposure:
  - mean first-window diffuse HT-like terms: 1.4737;
  - mean full-report diffuse HT-like terms: 1.4737;
  - no-diffuse first-window rate: 0.1053;
  - positive evidence exposed in first window: 0.9474.
- Stable TP comparison:
  - mean first-window diffuse HT-like terms: 1.6071.
- Evidence exposure strata:
  - diffuse evidence exposed in first window: 126 rows / 42 patients, FN rate 0.3492;
  - only generic morphology exposed: 12 rows / 4 patients, FN rate 0.2500;
  - no positive thyroid evidence exposed: 3 rows / 1 patient, FN rate 0.6667.
- Seed overlap:
  - seed 0 FN count: 22;
  - seed 42 FN count: 8;
  - seed 3407 FN count: 19;
  - all-seed FN intersection: 7 patients.

### C14-A Decision

- Final decision: `EVIDENCE_EXPOSED_BUT_NOT_USED`.
- C13 residual false negatives are not primarily explained by diffuse/HT-like evidence being beyond the model-visible text window.
- Do not proceed to a C14-B report-prefix pilot from this evidence.
- Next action should be analysis-first:
  - text representation audit;
  - patient-anchor fusion contribution audit;
  - image/text contribution audit;
  - seed-wise fusion stability audit.

## 2026-07-10 DMEA-v2 Phase C14-B Representation And Fusion Audit

### Plan

- Keep C14-B analysis-only and validation-only.
- Correct the C14-A cross-seed naming: `all_seed_fn`, `majority_fn`, `seed_sensitive_positive`, and `all_seed_tp`.
- Load the three selected C13 checkpoints in evaluation mode with `torch.no_grad()` and verify full-model predictions against the saved C13 validation predictions.
- Export available representation, anchor, classifier-contribution, evidence-role, and discordance diagnostics without changing the default model forward path.
- Run inference-only modality masking and C13 text occlusion diagnostics.
- Do not train, tune thresholds, modify labels/splits/task definition/manifests/report construction/model architecture, use test for route selection, or feed shortcut fields into the classifier.

### Intended Changes

- Add `scripts/analyze_phase_c14b_representation_fusion.py`.
- Add `scripts/collect_phase_c14b_report.py`.
- Generate reports under `analysis_reports/phase_c14b`.
- Record the exact validation-only route label and the allowed next-step class after server execution.

### Acceptance Checks

- Local `py_compile` passes for both C14-B scripts.
- All three C13 checkpoints load on the server.
- Full inference reproduces saved C13 validation probabilities within numerical tolerance.
- No optimizer is constructed or stepped, and no gradients are enabled.
- Missing internal diagnostics are reported rather than invented.
- C13 remains the current strict best; C14-B makes no model-improvement claim.

### Local Implementation Status

- Local `py_compile` passed for both C14-B scripts.
- Local commit: `a19f91b` (`Add C14b representation fusion audit`).
- GitHub push was attempted but timed out after 120 seconds.
- Bundle fallback was attempted, but the minimal read-only SSH probe to `linruixin@10.21.71.74:22` returned `Permission denied`; no server-side C14-B command was started.
- Therefore no C14-B route label, representation result, modality-ablation result, or text-occlusion result is claimed yet.

## 2026-07-10 DMEA-v2 Phase C14-B Multi-Seed Representation And Fusion Audit Revision

### Specification Corrections

- Adopted the revised multi-seed C14-B contract from `codex_dmea_ht_phase_c14b_multiseed_representation_fusion_audit.md`.
- Added a mandatory per-seed reproduction gate before masking, occlusion, or contribution claims.
- Added configurable `--seeds`, defaulting to `[0, 42, 3407]`.
- Corrected required output names for reproduction, representation, modality masking, text occlusion, and seed consistency artifacts.
- Added conservative route selection requiring valid reproduction and multi-seed directional support; otherwise the route is `MIXED_OR_INCONCLUSIVE`.
- Added explicit `unavailable` marking for learned fusion gate or attention values not exposed by the current model.

### Local Verification

- `python -m py_compile scripts/analyze_phase_c14b_representation_fusion.py scripts/collect_phase_c14b_report.py` passed.
- `git diff --check` passed.
- No training, optimizer, backward pass, label/split/task/manifest/tokenizer/report-construction/model changes were introduced.
- Server execution is still pending; no reproduction result or C14-B route claim is made in this section until the intended code is verified on the server.

### Server Synchronization And Execution

- Local revision commits: `7474777`, `939d4f4`, and `2f0bf7c`.
- GitHub synchronization succeeded; the server fast-forwarded to commit `2f0bf7c`.
- Server verification used `/home/linruixin/chen/conda/envs/ma/bin/python` and the exact C13 manifest/run directory specified for this phase.
- Required seeds `[0, 42, 3407]` were all loaded; no training, optimizer construction, backward pass, or test-based selection occurred.

### Reproduction Gate

- Each seed reproduced 94 / 94 saved validation predictions with matching patient IDs and labels.
- Per-seed maximum absolute probability difference: `1.1102230246251565e-16` for all three seeds.
- Per-seed mean absolute probability differences:
  - seed 0: `2.3400312420623313e-17`;
  - seed 42: `2.2588314197825658e-17`;
  - seed 3407: `2.048450062057719e-17`.
- The reproduction gate passed before contribution analysis.

### Corrected Cross-Seed Groups

- `all_seed_fn`: 7 patients.
- `majority_fn`: 12 patients.
- `seed_sensitive_positive`: 4 patients.
- `all_seed_tp`: 24 patients.

### Key Multi-Seed Findings

- Representation norms were close between all-seed FN and all-seed TP: text embedding norm `1.609` vs `1.649`.
- Text classifier contribution was lower for all-seed FN than all-seed TP: `-0.0028` vs `0.1082`; image contribution was more negative for FN: `-0.5146` vs `-0.2996`.
- Image masking raised all-seed FN probability in every seed: `+0.2582`, `+0.0285`, and `+0.0280`; the corresponding all-seed TP values were `+0.1887`, `+0.0031`, and `+0.0099`.
- Text-only-like rescue was not directionally stable for all-seed FN: `+0.3011`, `-0.0483`, and `-0.0156` across seeds 0, 42, and 3407.
- Bio masking was also not directionally stable for all-seed FN: `+0.0445`, `-0.0701`, and `-0.0263`.
- Removing diffuse clauses lowered all-seed FN probability in all seeds (`-0.0605`, `-0.0408`, `-0.1012`), confirming that visible text evidence affects prediction, but the effect was much weaker than for all-seed TP (`-0.3545`, `-0.3797`, `-0.5082`).
- Prefix-only deltas for all-seed FN were small and sign-inconsistent (`-0.0261`, `-0.0005`, `+0.0308`), while removing the C13 prefix lowered FN probability in all three seeds.
- The seed consistency audit therefore shows a mixed pattern: text evidence is represented and used, image masking suggests a reproducible suppression component, but text-only and bio effects are seed-sensitive and do not support one dominant mechanism.

### C14-B Final Decision

- Exact route label: `MIXED_OR_INCONCLUSIVE`.
- Allowed next-step class: `MORE_ANALYSIS_ONLY`.
- No new training pilot is authorized from C14-B.
- C13 remains the current strict best at validation AUC `0.8665 +/- 0.0077`; C14-B claims no model improvement.
- Test outputs were not used for route selection.

### Generated Server Artifacts

- `analysis_reports/phase_c14b/c14b_reproduction_check_by_seed.csv`
- `analysis_reports/phase_c14b/c14b_reproduction_check_report.md`
- `analysis_reports/phase_c14b/c14b_cross_seed_positive_groups.csv`
- `analysis_reports/phase_c14b/c14b_cross_seed_group_summary.csv`
- `analysis_reports/phase_c14b/c14b_representation_diagnostics_val.csv`
- `analysis_reports/phase_c14b/c14b_representation_group_summary.csv`
- `analysis_reports/phase_c14b/c14b_modality_masking_val.csv`
- `analysis_reports/phase_c14b/c14b_modality_masking_group_summary.csv`
- `analysis_reports/phase_c14b/c14b_modality_masking_seed_consistency.csv`
- `analysis_reports/phase_c14b/c14b_text_occlusion_val.csv`
- `analysis_reports/phase_c14b/c14b_text_occlusion_group_summary.csv`
- `analysis_reports/phase_c14b/c14b_text_occlusion_seed_consistency.csv`
- `analysis_reports/phase_c14b/c14b_seedwise_fusion_stability_val.csv`
- `analysis_reports/phase_c14b/c14b_seedwise_fusion_stability_report.md`
- `analysis_reports/phase_c14b/c14b_inputs_used_and_missing.csv`
- `analysis_reports/phase_c14b/phase_c14b_final_report.md`

## 2026-07-10 DMEA-v2 Phase C14-C To C15 Conditional Auto-Run

### Plan

- Run C14-C AUC pairwise ranking inversion decomposition before any training.
- Reproduce all three C13 validation checkpoints with the stricter C14-C thresholds: max absolute probability difference `<=1e-8`, mean absolute difference `<=1e-9`.
- Export all `47 x 47 = 2209` positive-negative validation pairs per seed, including full margins, diagnostic contribution margins, inference-only masking margins, and text occlusion margins.
- Assign overlapping image-opposed, text-driven, fusion-interaction, and hard-patient flags; aggregate all-seed, majority-seed, and single-seed inversion groups.
- Enter C15 only if the automatic route gate supports `IMAGE_DRIVEN_RANKING_FAILURE` or `FUSION_INTERACTION_RANKING_FAILURE` across seeds. Otherwise stop without training.
- Preserve the patient-level task, labels, splits, C13 manifest/report construction, shortcut exclusion, and test-as-reporting-only policy.

### Local Implementation

- Added `scripts/analyze_phase_c14c_pairwise_ranking.py`.
- Added `scripts/collect_phase_c14c_report.py`.
- Local `py_compile` and `git diff --check` passed.

### C14-D Server Result

- Server synchronized to `17da324` through the bundle fallback after a transient GitHub HTTP/2 pull error.
- C14-D completed as an audit-only run; no training started.
- All-seed hard patients: `79` total, comprising `43` negative and `36` positive patients.
- Top-20 hard-patient incidence share: `66.27%`, using the patient-side incidence denominator `2 x inversion rows`.
- Negative hard patients had mean image-opposed rate `0.5868` and mean image-repair rate `0.2406`.
- Positive hard patients had mean text-driven rate `0.6841` and mean image-opposed rate `0.1873`.
- Mean fusion-interaction rates remained low: `0.0704` for negative hard patients and `0.1255` for positive hard patients.
- C14-D gate: `HARD_PATIENT_SUBGROUP_AUDIT_CONFIRMED`.
- Next step remains `MORE_ANALYSIS_ONLY`; C15 remains unauthorized.

## 2026-07-10 DMEA-v2 Phase C14-E Hard Clinical Evidence Audit

### Plan

- Reconstruct hard-positive, hard-negative, and same-label non-hard validation controls from C14-C/C14-D.
- Standardize top-5/top-10/top-20 responsibility using separate pair-coverage, patient-side incidence, and unique-pair denominators.
- Match controls without replacement using only report length, selected visits, used images, image padding, bio availability, and bio missingness.
- Audit hard-positive HT-specific/generic/contradictory text evidence, visit-level temporal states, and C14-B multimodal diagnostics.
- Audit hard-negative image-mimic categories, follow-up/label-boundary uncertainty, and multimodal support without inferring diagnoses or changing labels.
- Report patient-level effect sizes and bootstrap intervals, then apply the 30% generalizability gate.
- Keep C15 blocked; no C14-E route automatically authorizes training.

### Local Implementation

- Added `scripts/analyze_phase_c14e_hard_clinical_evidence.py`.
- Added `scripts/collect_phase_c14e_report.py`.
- Local `py_compile` and `git diff --check` passed.
- No core model, training code, manifest, label, split, task, report, image, or bio changes were made.

### C14-E Server Result

- Server synchronized to `e9a0486` through the bundle workflow and completed the C14-E audit.
- Hard positives / non-hard positives: `36 / 11`.
- Hard negatives / non-hard negatives: `43 / 4`.
- No-replacement matching retained `11` hard positives and `4` hard negatives, for coverage `30.56%` and `9.30%`; `25` hard positives and `39` hard negatives remained unmatched.
- No matching variable reached the preferred `|SMD| <= 0.10` balance after matching. Post-match absolute SMDs ranged from `0.5141` to `1.1146` for positives and `0.4629` to `1.1078` for negatives, excluding unavailable/constant bio-availability values.
- Corrected all-patient top-k metrics:
  - top-5 all-seed pair coverage / patient-side incidence share: `70.23% / 36.51%`;
  - top-10 all-seed pair coverage / patient-side incidence share: `92.09% / 50.93%`;
  - top-20 all-seed pair coverage / patient-side incidence share: `100.00% / 71.40%`.
- Hard-positive evidence categories included `27/36` HT-specific cases, but `21/36` also had contradictory evidence; temporal states included `26/36` intermittent-conflict cases.
- Hard-negative image-mimic audit found only `2/43` strict model-supported mimic cases, while `17/43` had thyroiditis-like report wording without sufficient image-support evidence and `24/43` remained unclear.
- Hard-negative follow-up/label audit found `25/43` HT-like-but-not-diagnosed, `5/43` short-follow-up/uncertain, and `13/43` well-supported negatives.
- Candidate coverage before matched-control gating:
  - hard-positive weak/ambiguous or temporal-conflict evidence: `30/36`;
  - hard-negative strict image mimic: `2/43`;
  - label/follow-up ambiguity: `30/43`.
- All candidate mechanisms failed the generalizability gate because matching coverage and balance were inadequate and matched-control contrasts did not support specificity.
- Final route: `DATA_LIMIT_NO_GENERAL_MODEL_FIX`.
- Allowed next step: `KEEP_C13_AND_REPORT_LIMITATION`.
- Training remains blocked; C13 remains the current strict best.

## 2026-07-10 DMEA-HT Final C13 Reproducible Delivery

### Plan

- Freeze the C13 temporal-focus route as the final strict best under the available evidence.
- Verify the three validation-selected checkpoints, checkpoint metadata, manifest invariants, prediction row counts, C14-B reproduction gate, validation/test metrics, shortcut residuals, and C14-E training stop.
- Generate checkpoint/manifest/report SHA256 inventory, server environment capture, model card, exact reproducibility commands, and final delivery report.
- Keep the formal claim at the three-seed single-model mean; do not test-select a seed or claim an ensemble.
- Report that validation AUC 0.90 was not reached and that C14-E found a data-limit/no-general-fix route.

### Local Implementation

- Added `scripts/collect_final_c13_delivery.py`.
- Local `py_compile` and `git diff --check` passed.
- No training, model, data, label, split, task, manifest, report construction, image, bio, or threshold changes were made.

### C14-C Server Result

- Server synchronized to commit `238227a` and ran the C14-C pairwise audit in the `ma` environment with CUDA.
- Reproduction and pairwise analysis completed for seeds `[0, 42, 3407]`.
- The audit produced `2209` positive-negative pairs per seed and `885` inversion rows across the three seeds.
- Cross-seed aggregation found `215` all-seed inversion pairs.
- Route: `HARD_PATIENT_SUBGROUP_FAILURE`.
- Final status: `C14C_HARD_SUBGROUP_STOP`.
- Allowed next step: `MORE_ANALYSIS_ONLY`.
- C15 was not authorized and no training process was started.
- Server artifacts are under `analysis_reports/phase_c14c/`; local retrieval was deferred after the tool usage limit was reached.

### C14-C Detailed Gate Review

- The complete pairwise table contains `6627 = 2209 x 3` patient pairs.
- Total inversion rows: `885`.
- All-seed inversion pairs: `215`.
- Majority-seed inversion pairs: `75`.
- Single-seed inversion pairs: `90`.
- Top-five patient inversion share: `59.32%`, with multiple all-seed hard patients; this satisfies the hard-subgroup concentration stop condition.
- Image-opposed fraction among stable inversion rows: `28.55%`, below the `30%` image route threshold.
- Image-masking repair-positive seed count: `0`; image masking did not repair the inversion margin in any formal seed.
- Text-driven fraction among stable inversion rows: `62.89%`, but the concentration and route gate rules still select the hard-patient stop rather than a general text route.
- Fusion-interaction fraction: `8.55%`, below the route threshold.
- C15 conflict-gated training remains unauthorized. The next valid action is a narrower hard-patient/subgroup audit, not training.

## 2026-07-10 DMEA-v2 Phase C14-D Hard-Patient Subgroup Audit

### Plan

- Follow the C14-C `HARD_PATIENT_SUBGROUP_FAILURE` stop with a narrow analysis-only subgroup audit.
- Profile all patients that produce inversion rows in all three seeds, with a top-k impact table.
- Compare hard positive/negative cohorts with non-hard validation patients using C14-B representations, classifier diagnostics, C14-A evidence exposure fields, report metadata, and inversion flags as audit-only fields.
- Do not enter C15, train a new model, tune thresholds, change labels/splits/task/manifest/report construction, or feed audit fields into a predictor.

### Local Implementation

- Added `scripts/analyze_phase_c14d_hard_patient_audit.py`.
- Added `scripts/collect_phase_c14d_report.py`.
- Local `py_compile` and `git diff --check` passed.

## 2026-07-10 Final C13 Delivery Status

- Froze `C13_TEMPORAL_FOCUS_DMEA_HT` as the strict-best route under the completed C13-C14E gates.
- Added the frozen route summary to `README.md` and created `FINAL_MODEL_SELECTION.md` with the exact contract, three-seed validation metrics, shortcut audit, hard-subgroup limitation, checkpoint paths, and verifier command.
- The formal claim remains the mean of seeds `[0, 42, 3407]` selected by validation AUC; test metrics remain reporting-only.
- C15 remains blocked because C14-E returned `DATA_LIMIT_NO_GENERAL_MODEL_FIX`; no additional training was launched.
- The final server verifier is not yet declared complete. Two bundle upload attempts timed out while connecting to `10.21.71.74:22`; server execution and artifact retrieval remain required before closing the delivery goal.

## 2026-07-13 Final C13 Delivery Completion

- Server connectivity recovered and the repository was fast-forwarded from GitHub without modifying the 39 historical untracked scripts, logs, or backup directories.
- Ran `scripts/collect_final_c13_delivery.py` in `/home/linruixin/chen/conda/envs/ma/bin/python` on server `5090-01`; no training was started.
- Corrected the Markdown serializer to preserve integer seed and epoch fields, synchronized commit `00cf4f0`, and reran the complete verifier.
- Final verifier result: `delivery_pass=true`; all `16/16` delivery checks passed and all `16/16` inventory entries existed with SHA256 hashes.
- Verified manifest size `780` with frozen patient-level counts: train `301/301`, validation `47/47`, and test `42/42` for labels `0/1`.
- Verified all three checkpoints load with matching seed and best-epoch metadata, validation prediction rows `94` per seed, test reporting-only rows `84` per seed, and C14-B reproduction maximum absolute probability difference `1.11e-16` per seed.
- Frozen validation AUC: `0.8664554096876415 +/- 0.0077356303714961`; shortcut safety and the C14-E training block both passed verification.
- Final decision: `FREEZE_C13_AS_STRICT_BEST_AND_REPORT_LIMITATION`; C15 remains unauthorized and the validation AUC 0.90 target was not reached.
- Retrieved the 13-file final package to `analysis_reports/final_c13_delivery/` and independently confirmed zero failed checks and zero missing inventory artifacts locally.

## 2026-07-13 PowerShell And SSH Transport Stabilization

- Diagnosed Windows PowerShell 5.1 with code page `936`, console input `GB2312`, native pipeline encoding `US-ASCII`, and a pre-existing Conda `profile.ps1` blocked by the default `Restricted` execution policy.
- Did not retain a broader PowerShell execution-policy change; restored `CurrentUser` to its original `Undefined` value.
- Added `scripts/invoke_remote_bash.py`, which uses Python standard-library `subprocess` with `shell=False` and sends validated UTF-8 Bash files directly to SSH stdin.
- This transport prevents PowerShell from expanding remote Bash expressions such as `$(pwd)` and awk `$3`, and avoids PowerShell native-pipeline transcoding.
- Verified the transport against server `5090-01`: Chinese text round-tripped correctly, remote `$(pwd)` evaluated on Linux, and awk `$3` returned the expected value.
- Codex-side PowerShell commands should use a non-login shell so the blocked Conda profile is not loaded; server commands should use the Python transport rather than embedded SSH one-liners.

## 2026-07-13 Phase C16 Disease-State Anchored Shared-Specific Alignment

### Baseline And Authorization

- Started from clean commit `b91bd1d` on branch `feature/c16-disease-state-shared-specific-alignment`.
- C13 temporal-focus DMEA-HT remains the frozen strict-best fallback: validation AUC `0.8664554097 +/- 0.0077356304`, formal seeds `[0, 42, 3407]`.
- C16 is an explicitly authorized, independent disease-state alignment hypothesis. It does not reopen C15 or the C14-E local patch routes.
- Keep the C13 manifest, patient IDs, labels, patient-level split, image paths, bio values, prediction horizon, report construction, encoders, optimizer family, and validation-AUC checkpoint rule unchanged.
- Test remains reporting-only and cannot select the architecture, loss, checkpoint, threshold, fallback, or promotion decision.

### Implementation Plan

- Add an optional DSSA module that is instantiated only when `model.use_dssa=true`; the disabled path must preserve legacy module construction and forward behavior.
- Reuse the C13 image, text, and bio encoder global outputs. For each modality, learn lightweight shared and specific projections; normalize only the shared representation.
- Learn two normalized disease-state prototypes initialized deterministically after the training seed is set. Compute per-modality prototype logits for training-label CE, but never pass labels into model inference.
- Align only same-patient shared components. Keep specific components complementary with shared-specific orthogonality, batch variance protection, bounded residual scale `rho=0.10`, near-zero residual initialization, and validity-mask-aware attention/gates.
- Form the C16 representation from patient shared state, soft predicted disease anchor, controlled specific residual, and prototype disease margin. Shortcut/audit fields are never read by the module or losses.
- Add training-only positive-negative batch ranking with safe zero return for one-class batches. BCE remains dominant.
- Use three classification-only warmup epochs, ramp DSSA/ranking weights to their fixed targets through epoch 8, then keep them fixed. Log all raw losses, effective weights, prototype/attention/gate/norm health, validation metrics, and selection state.
- Export selected-checkpoint patient diagnostics and pairwise validation ranking, then generate prototype, shared, specific, positive-preservation, inversion, shortcut, seed-stability, comparison, and final gate reports.

### Execution Gate

- Run static/synthetic checks and a 1-2 epoch seed-0 smoke first. Stop on non-finite values, constant predictions, prototype or sample collapse, global attention/gate saturation, residual domination, legacy incompatibility, or shortcut leakage.
- Run the full seed-0 pilot only after smoke passes. Compare against C13 seed-0 validation AUC `0.8655500226` and all documented performance/alignment/safety gates.
- If seed 0 fails, permit exactly one fallback with `lambda_rank=0`; all other architecture and training settings remain fixed.
- Run seeds `42` and `3407` only after a passing seed-0 route. Do not tune after seeing seed 0.
- Promote C16 only if every formal performance, alignment, stability, positive-preservation, inversion, and shortcut gate passes. Otherwise keep C13 and use exactly one documented C16 decision label.

### Static And Synthetic Result

- Local `py_compile` and `git diff --check` passed for the DSSA module, legacy model integration, training loop, and C16 audit/report scripts.
- Server `5090-01` synchronized the feature branch in the `ma` environment and completed the synthetic smoke.
- Expanded synthetic/config-contract result: `28/28 PASS`.
- Legacy absent-vs-explicit-disabled DSSA state-dict keys matched (`65` keys) and logits were exactly equal (`max_abs_difference=0.0`).
- C16 output shape, all floating outputs, masked attention sums, and missing-modality behavior passed.
- Ranking loss was finite for a mixed batch and returned graph-connected zero for all-positive and all-negative batches.
- All six raw DSSA/ranking losses were finite. Nonzero finite gradients reached shared projectors, specific projectors, prototypes, shared attention score, specific gates, specific residual projectors, and the C16 classifier.
- Initial prototype cosine was `0.08491625`; maximum scaled specific-residual/shared ratio was `0.00127990`; no shortcut field was present in the alignment module.
- The expanded gate also verified the frozen C13 data/manifest, encoder/input settings, pilot/stress optimizer and epoch budget, formal seed partition, fixed DSSA loss weights, disabled legacy weak-label losses, bounded residual scale, and validation-AUC checkpoint contract.
- The first real 2-epoch smoke attempt stopped before training because the only visible RTX 5090 was occupied by another user's process (`12072 MiB`). No DMEA training process was launched, no process was interrupted, and no GPU polling was started.
