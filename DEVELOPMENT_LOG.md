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

### Server Worktree And Reporting Audit

- A second one-shot GPU check found the same external process still using `12072 MiB`; C16 training remained stopped and no polling process was created.
- The original server worktree contained only the two synthetic reports regenerated by this task as tracked modifications. Their Git blob hashes exactly matched the incoming committed versions, but the worktree was left untouched in accordance with the no-reset/no-stash rule.
- Created a clean detached server worktree at `/home/linruixin/chen/project/DMEA-HT-c16` from commit `282c95f`. Server `py_compile` and CLI import/help checks passed there.
- Expanded the C16 collector to record commit/worktree/server environment, validation and reporting-only test metrics, seed-wise C13 deltas, inversion deltas, positive-preservation and all-seed FN/TP/FP changes, prototype/shared/specific health, shortcut safety, target-AUC status, final decision label, and the current strict best.
- Strengthened the health gate with disease-margin direction, prototype assignment, selected-epoch shared-consistency improvement, gate/attention distributions, and explicit collapse/duplication/dominance flags.

### GPU Blocked Audit

- A third consecutive goal turn found the same external process (`PID 89376`) using `12072 MiB` on the only visible RTX 5090.
- The clean C16 worktree remained unchanged at `282c95f`; no smoke, pilot, fallback, or stress training process was started.
- Static/config/synthetic work is complete (`28/28 PASS`), but the required real 2-epoch smoke cannot be replaced by a narrower CPU or synthetic result.
- Phase C16 is formally blocked at the real-smoke gate until the GPU becomes available. The goal must be resumed from the smoke step; seed-0 pilot remains unauthorized until smoke and health checks pass.

## 2026-07-13 Urgent C16 DSSA Correction

- The prior C16 DSSA/shared-specific plan was invalid for this project and is superseded by this correction.
- One seed-0 two-epoch smoke had completed and produced a checkpoint before the correction. The full seed-0 pilot and stress seeds were never started.
- No C16 process was alive when the correction was applied. The isolated server smoke and audit directories were preserved and marked `ABORTED_MISGUIDED_C16`, `NOT_FOR_MODEL_SELECTION`, and `NOT_FOR_REPORTING`.
- The mistaken C16 commits were never merged into `main`; both `main` and `origin/main` remained at the verified pre-C16 base `b91bd1d`.
- The complete mistaken branch delta was preserved in `analysis_reports/c16_correction/mistaken_c16_changes.patch` before reverting.
- `dmea_ht/models.py` and `train.py` were restored exactly to the pre-C16 base. DSSA-only alignment, config, audit, collector, and synthetic-report files were removed from the corrected feature-branch state.
- No result from the mistaken C16 branch is valid for model selection, comparison, promotion, or scientific reporting.
- The project main line remains disease-mechanism and evidence-aware multimodal alignment for next-year HT prediction; it is not related to DecAlign and must not inherit shared/private decomposition concepts or terminology.
- C13 temporal-focus remains intact and is the current strict-best baseline with mean validation AUC `0.8664554097` over seeds `[0, 42, 3407]`.
- The corrected next phase is C16-MEA. It may begin with a design audit only; no new implementation or training is authorized until that audit is complete and reviewed.
- Correction status: `MISGUIDED_C16_STOPPED_AND_REVERTED`.

## 2026-07-13 Phase C16-MEA Mandatory Design Audit

### Authorization And Boundary

- Started a new audit-only branch `codex/c16-mea-design-audit` from corrected commit `3363e98`; the abandoned DSSA branch remains preserved for audit and is not the C16-MEA work branch.
- C16-MEA is the corrected disease-mechanism and evidence-aware phase. It must not reintroduce shared/private decomposition, DecAlign terminology, generic modality-invariant alignment, or shared-specific orthogonality losses.
- The current task is the mandatory design audit only. No model implementation or training is authorized until the audit establishes a valid path from real fields and existing diagnostics.
- C13 temporal-focus remains the frozen strict-best baseline. Patient IDs, labels, patient-level splits, prediction horizon, manifest, report construction, image paths, and bio values remain unchanged; test remains reporting-only.

### Pre-Implementation Findings

- C13 exposes per-image tokens and an image global embedding, per-character text tokens and a pooled text embedding, seven bio tokens and a bio global embedding, the patient anchor, evidence scores, modality classifier contributions, and discordance norms.
- The fixed bio order in the manifest builders is `sex, age, TgAb, FT3, FT4, TPOAb, TSH`. TgAb/TPOAb and FT3/FT4/TSH may be grouped semantically only after the server audit confirms the real source-table and manifest fields.
- Existing `bio_abnormal_flags` are zero-filled placeholders unless an explicit trusted source is present. No reference range or abnormal direction may be invented; if trust cannot be established, bio remains observed continuous evidence with validity masking only.
- Existing text dictionaries cover morphology, diffuse HT-like wording, strong/weak normal or opposing wording, uncertainty, benign/nodular morphology, and diagnostic hints. They may guide token pooling but may not become patient targets or revive weak-label BCE.
- C13 text includes explicit `[C13_LATEST_THYROID ...]`, `[C13_HISTORY_THYROID]`, and `[C13_FULL_REPORT]` markers when a focus prefix is available. The real-manifest audit must quantify marker coverage and whether latest/history segments can be reconstructed inside the model-visible character window.
- C14-A exposure, C14-B representation/masking/occlusion, C14-C pairwise inversion, C14-D hard-patient, and C14-E matched-control outputs are reusable only as diagnostics. Shortcut and structural fields remain audit-only.

### Audit Plan

- Add `scripts/audit_phase_c16_mea_design_inputs.py` to inspect the real C13 manifest, source-table schema, current model/data paths, existing text dictionaries, temporal markers, C14 artifacts, masks, and shortcut exclusions without changing data.
- Add `scripts/collect_phase_c16_mea_design_report.py` to generate and validate the nine required design-audit deliverables under `analysis_reports/phase_c16_mea_design/`.
- Run static checks locally, then execute the audit on the server with `/home/linruixin/chen/conda/envs/ma/bin/python` against `/data/csb/DMEA-HT/HT_2025.12_25/manifest_distmatch_structmatch_evidence_v2_c13_temporal_focus.jsonl` and `all_patients.xlsx`.
- Do not begin C16-MEA model coding or training unless the collected report records a feasible, shortcut-safe implementation path with explicit limitations.

### Real-Data Design Audit Result

- Added `scripts/audit_phase_c16_mea_design_inputs.py` and `scripts/collect_phase_c16_mea_design_report.py`. The audit reads constants by AST and does not import the torch training stack; local `py_compile`, CLI checks, and a two-row synthetic nine-file delivery test passed.
- Created the isolated server worktree `/home/linruixin/chen/project/DMEA-HT-c16-mea` and ran the audit with `/home/linruixin/chen/conda/envs/ma/bin/python` against the frozen C13 manifest and `all_patients.xlsx`; no GPU or training process was used.
- Real manifest verification passed for `780` unique patients with unchanged counts: train `301/301`, validation `47/47`, and reporting-only test `42/42` for labels `0/1`.
- All `780` rows contain seven bio values, seven missing-mask entries, and seven abnormal-flag entries in the verified order `sex, age, TgAb, FT3, FT4, TPOAb, TSH`. All seven source-table columns exist.
- `sex`, `age`, `FT3`, `FT4`, and `TSH` are observed in all manifest rows. `TgAb` is observed in `5.64%` and `TPOAb` in `5.38%`; subgroup availability is exported by split and label.
- No reference-range columns or trusted abnormal metadata exist, and every stored abnormal flag is zero. Bio fields may be grouped as observed continuous semantics, but no abnormal/normal/support/opposition target or rule is valid. Sparse antibody availability must not become evidence about whether a test was ordered.
- The C13 latest marker is present and model-visible in `766/780` rows (`98.21%`), the history marker in `422/780` (`54.10%`), and the full-report marker in `771/780` (`98.85%`); `417` rows contain both latest and history markers. Missing sections require learned fallback pooling and cannot be treated as negative evidence.
- Audited `81` existing dictionary entries across morphology, diffuse HT-like, opposition/normal, uncertainty, diagnostic hint, and benign/nodular groups. They are authorized only as context-aware character-position pooling masks, never as patient targets or weak-label BCE supervision.
- All `8/8` required C14 diagnostic artifacts were available. Their representation, masking, occlusion, inversion, hard-patient, and matched-control fields remain audit-only and cannot enter the predictor.
- The shortcut exclusion map covers all required structural variables plus C13 focus counts and selected visit dates. Only image, text, and per-field bio validity masks may enter computation, without predictive missingness/count scalars.
- Design gate: `C16_MEA_DESIGN_AUDIT_PASS_WITH_CONSTRAINTS`; all `8/8` hard checks passed and all nine required design files were generated under `analysis_reports/phase_c16_mea_design/`.
- The first permitted mechanism-alignment loss is image-text morphology alignment only. Bio immune/function nodes may enter the mechanism graph from verified observed fields, but no bio-text alignment is permitted without separately verified matching text semantics.
- Backward-compatible C16-MEA implementation may begin. Training remains blocked until static/synthetic checks and both predefined seed-0 Core and Core+Ranking smoke gates pass.

## 2026-07-13 Phase C16-MEA Implementation

### Pre-Edit Contract

- Implement C16-MEA on a new branch `codex/c16-mea-implementation` from the passing design-audit commit `b04e0d2`; do not merge or reuse the abandoned DSSA implementation.
- Keep the frozen C13 manifest, labels, patient-level splits, report construction, image paths, bio values, prediction horizon, encoder families, optimizer family, and validation-AUC checkpoint selection unchanged. Test remains reporting-only.
- Reuse the C13 per-image, per-character text, and seven-field bio encoder tokens. Add only an optional `model.use_mea=true` path; absent and explicit-false configurations must retain identical legacy state dictionaries and logits.
- Build text role masks from audited morphology, support, opposition, uncertainty, nonspecific, latest, history, and full-report character spans. Empty masks use learned token pooling; missing sections are never negative evidence.
- Group bio tokens only as verified observed continuous semantics: demographics (`sex`, `age`), immune (`TgAb`, `TPOAb`), and thyroid function (`FT3`, `FT4`, `TSH`). Do not read `bio_abnormal_flags`, test-order missingness counts, shortcut variables, or C14 audit fields.
- Form six explicit mechanism nodes for morphology, immune evidence, function evidence, opposition, temporal evidence, and disease state. Use only audited graph edges and conflict-aware evidence aggregation; do not introduce shared/private decomposition, modality-invariant alignment, orthogonality, or DecAlign terminology.
- Permit image-text morphology alignment as the only cross-modal mechanism-alignment loss. Add a disease-state classification margin, evidence-role separation, and optional training-batch positive-negative ranking; BCE remains dominant and labels never enter model inference.
- Use classification-only epochs `1-3`, linearly ramp auxiliary weights through epoch `8`, and then hold fixed targets. Core uses `lambda_rank=0`; Core+Ranking uses `lambda_rank=0.02`. All architecture, data, and remaining loss settings stay identical between the two routes.
- Log raw and effective losses plus mechanism, role, reliability, conflict, state-margin, attention, validity, and prediction diagnostics. Export only patient-level scalar diagnostics, not raw token tensors.

### Execution Gates

- Before any real training, require local static checks and a server synthetic gate covering legacy equivalence, tensor shapes, masks and empty-mask fallback, missing bio groups, finite losses, ranking one-class behavior, probability normalization, gradient reachability, and shortcut/DSSA exclusion.
- After the static/synthetic gate passes, run predefined two-epoch seed-0 Core and Core+Ranking smokes. Stop on non-finite values, constant predictions, invalid probability sums, empty gradients, global attention/reliability saturation, conflict collapse, or legacy incompatibility.
- Full seed-0 pilots are authorized only after both smoke routes and health collectors pass. Select the route using the frozen validation-only gate; do not inspect reporting-only test metrics for route or checkpoint selection.
- Seeds `42` and `3407` are authorized only after a passing seed-0 route. No architecture or hyperparameter tuning is allowed after seed-0 results are observed.

### Local Implementation And Synthetic Gate

- Added the optional `MechanismEvidenceAlignment` path over unchanged C13 image, text, and bio encoder tokens, with image morphology queries, mask-guided text role pooling, observed-only bio grouping, six HT mechanism states, latent evidence-role scoring, conflict-aware aggregation, and a binary disease-state head.
- Separated text evidence availability from dictionary-guidance availability. Empty support/opposition/uncertainty/temporal masks use learned token pooling, while image-text morphology alignment is enabled only for a real audited morphology span and a valid image pair. Diagnostic hints may guide support pooling but cannot authorize morphology alignment.
- Added patient-label state margin, image-text morphology alignment, clinical support/opposition separation, and training-batch-only pairwise ranking losses. Effective auxiliary weights are zero for epochs `1-3`, ramp linearly through epoch `8`, and remain fixed thereafter.
- Added Core and Core+Ranking smoke/pilot configs plus a formal multi-seed config. Seed-0 smoke/pilot configs set `evaluate_test=false`; only the post-selection formal config reports test.
- Added scalar MEA diagnostics to epoch and selected-checkpoint prediction exports, plus separate health, seed-0 route-selection, formal comparison, inversion, evidence-role, mechanism, temporal, positive-preservation, shortcut, seed-stability, and final-decision collectors.
- Local coding checks parsed all `31` YAML configs, compiled the target Python files, and found no whitespace errors. These are coding-only checks and are not accepted as runtime evidence.
- A local CPU synthetic preflight was mistakenly invoked despite the server-only execution contract. No training was started, its generated report was deleted, and its result is invalid for authorization, comparison, or reporting.
- All synthetic and runtime checks must be rerun in `/home/linruixin/chen/conda/envs/ma` on the server. Neither predefined seed-0 smoke is authorized until that server gate passes.

### Server Static And Synthetic Gate

- GitHub fetch from the server failed with `GnuTLS recv error (-110)`. The already pushed branch was transferred as a Git bundle without changing commit identity, and a new detached implementation worktree was created at `/home/linruixin/chen/project/DMEA-HT-c16-mea-impl`.
- The original design-audit worktree and its untracked generated reports were left untouched. The implementation worktree is pinned to commit `1154dec8208fa273680676f109c77564ffc16bae`.
- Ran all static and synthetic checks with `/home/linruixin/chen/conda/envs/ma/bin/python` and `CUDA_VISIBLE_DEVICES` empty. Server result: `21/21 PASS`.
- Legacy absent-versus-explicit-false MEA state dictionaries matched across `65` keys and logits matched exactly (`max_abs_difference=0.0`). MEA output shape, `14 x 3` role probabilities, finite values, empty-mask fallback, missing bio groups, all-missing mechanism graph, probability normalization, and warmup/ramp checks passed.
- Mixed-batch ranking was finite; all-positive ranking returned a graph-connected zero. Nonzero gradients reached image, text, bio, role scorer, mechanism relation, conflict aggregator, and disease-state head modules.
- Model-source checks found none of the prohibited shortcut inputs or DSSA/shared-specific terms. Server evidence is stored as `analysis_reports/phase_c16_mea/c16_mea_synthetic_gate_server.json`.
- Server static/synthetic status: `PASS`. The two predefined seed-0 smoke runs are now authorized; seed-0 full pilots remain blocked until both smoke health gates pass.

### Server Smoke Launch Status

- Prepared a server-only sequential driver for Core smoke, Core health audit, Rank smoke, and Rank health audit. All four C16 smoke/pilot configs disable test evaluation; the driver never reads reporting-only test data.
- Performed one pre-launch GPU check without polling. Available GPU memory was `8537 MiB`, below the project-scoped `12000 MiB` shared-server safety threshold while another workload was active.
- The driver exited before launch. No C16-MEA training process was started, no existing process was interrupted, and no GPU polling job was created.
- Resume from the same smoke launch command when sufficient GPU memory is available. Full seed-0 pilots remain unauthorized until both smoke runs and their validation-only health gates pass.

### Server Smoke Launch

- At the user's explicit launch instruction, repeated the single pre-launch check and found `8537 MiB` GPU memory free. For these lightweight two-epoch smoke runs, the launch floor was reduced to `6000 MiB`; no architecture, batch size, loss, data, or training setting changed.
- Started the server-only sequential smoke driver at `2026-07-13T17:22:46+08:00` with PID `361001` from implementation commit `a973bedb3d3c340dfd905fc3f985f9ad905fc88d`.
- Driver order is fixed: Core smoke -> Core validation health gate -> Rank smoke -> Rank validation health gate. Rank cannot start if Core health fails. The two routes never run concurrently.
- Launch status was `RUNNING`. No polling process was created; completion must be checked only on a later explicit status request or completion notification.

### Server Smoke Completion

- The sequential smoke driver completed with `status=PASS` at `2026-07-13T17:24:28+08:00`; its process exited normally after both routes finished.
- Core smoke and Core health gate passed with validation AUC `0.8506111363`, AUPRC `0.7966925771`, sensitivity `0.8297872340`, specificity `0.7234042553`, and `94` validation prediction rows.
- Rank smoke and Rank health gate passed with validation AUC `0.8506111363`, AUPRC `0.7966925771`, sensitivity `0.8297872340`, specificity `0.7234042553`, and `94` validation prediction rows.
- Both health reports passed `15/15`; all prediction diagnostics were finite, prediction standard deviation was `0.1282`, role probabilities were normalized, conflict was not saturated, modality weights were not saturated, mechanism norms were bounded, and evidence roles did not collapse.
- Smoke configs generated no test predictions. Rank weight remained effectively zero during the two-epoch warmup, so identical Core/Rank smoke metrics are expected and are not a route selection result.
- Both predefined seed-0 full pilots are now authorized. No architecture or loss tuning is permitted between them, and test remains disabled until formal route selection.

### Seed-0 Pilot Launch

- After the two smoke routes and both `15/15` health gates passed, started the fixed seed-0 pilot driver on the server at `2026-07-13T17:28:57+08:00`.
- Server driver PID: `376289`; implementation worktree commit: `a973bedb3d3c340dfd905fc3f985f9ad905fc88d`.
- Fixed order is Core seed-0 pilot -> Core validation health gate -> Rank seed-0 pilot -> Rank validation health gate. No test predictions are generated during this stage, and no route or hyperparameter changes are allowed.
- Launch status: `RUNNING`. No polling process was created.

### Seed-0 Pilot Completion And Gate Decision

- The seed-0 pilot driver completed normally with `status=PASS` at `2026-07-13T17:36:37+08:00`; Core and Rank checkpoints were produced and both post-run health gates passed `15/15`.
- Core seed-0: best epoch `3`, validation AUC `0.8764146673`, AUPRC `0.8337014838`, sensitivity `0.2978723404`, specificity `0.9574468085`, positive-negative gap `0.2253515064`, pairwise inversions `273` versus C13 `297`.
- Rank seed-0: best epoch `3`, validation AUC `0.8759619737`, AUPRC `0.8266092853`, sensitivity `0.2978723404`, specificity `0.9574468085`, positive-negative gap `0.2251248348`, pairwise inversions `274` versus C13 `297`.
- The validation-only route collector did not read test and returned exactly `C16_MEA_PILOT_FAIL_KEEP_C13` for both routes. Core and Rank failed the fixed pilot gate because AUPRC decreased by more than `0.005`, sensitivity fell below `0.55`, the positive-negative gap materially decreased, and positive probabilities showed global suppression.
- C16-MEA therefore did not enter formal multi-seed evaluation. C13 temporal-focus remains the current strict-best and fallback; no additional C16 tuning or rescue run is authorized in this phase.
- The server route decision artifacts are under `analysis_reports/phase_c16_mea/`; test remains ungenerated and reporting-only.

## 2026-07-13 Phase C17 DEMA-HT Residual Refinement

### Pre-Edit Contract

- Official model and research name: `DEMA-HT` (`Disease-Mechanism and Evidence-Aware Multimodal Alignment for Hashimoto's Thyroiditis Prediction`). Historical repository/package identifiers `DMEA-HT` and `dmea_ht` remain unchanged for reproducibility.
- A terminology correction is recorded after C16-MEA. The alignment axis is HT pathological mechanism; image, report-text, and biochemical evidence are the aligned objects; HT/non-HT is only the final binary prediction target. The correct description is: "align multimodal clinical evidence through HT pathological-mechanism relations and aggregate the mechanism evidence for HT risk prediction." This does not invalidate C16 automatically; the computation graph remains the basis for assessment.
- Freeze the verified C13 temporal-focus checkpoint, manifest, labels, patient-level split, history cutoff, report construction, image paths, bio values, encoder family, optimizer family, epochs, and validation-AUC checkpoint selection. Do not use saved predictions as training inputs. Test remains reporting-only and is disabled for smoke and seed-0 pilots.
- C17 is `DEMA-HT Pathological-Mechanism Evidence Residual Refinement and Positive-Evidence Preservation`. Reuse the completed DEMA evidence projectors, mechanism relation layer, conflict aggregator, and mechanism aggregation head. Do not add projectors, graph nodes, modality branches, shared/private decomposition, generic alignment, ranking loss, or shortcut variables.
- Add only a bounded residual correction: `raw_delta = MLP(h_mechanism_correction)`, `delta_logit = 0.50 * tanh(raw_delta)`, and `final_logit = base_logit + delta_logit`. The residual output layer is zero-initialized, so pretraining equivalence must satisfy `max_abs_logit_difference <= 1e-8`.
- Run two fixed variants: DEMA-R BCE with `0.001 * mean(delta_logit^2)`, and DEMA-RP with the additional positive-preservation penalty `0.02 * relu(-delta_positive - 0.05)`. All C16 auxiliary and ranking weights are zero. The all-negative positive-preservation term must remain graph-connected zero.
- Only C17 validation AUC is allowed for checkpoint selection, route comparison, promotion, rejection, and formal authorization. C17 reports, tables, gates, and final decision files must not contain the forbidden secondary ranking metric. Sensitivity, specificity, balanced accuracy, positive-negative gap, inversion count, residual diagnostics, mechanism diagnostics, and shortcut audits are safety diagnostics only.
- Seed-0 pilot authorization requires the server static/synthetic gate and both smoke health gates. Formal seeds `[0, 42, 3407]` are authorized only after a seed-0 route passes the fixed residual gate. If no route passes, retain C13 and record `DEMA_C17_PILOT_FAIL_KEEP_C13`.

### Planned C17 Gates

- Required server-only checks: legacy checkpoint compatibility, C16 head rename logit equivalence, zero-residual equivalence, frozen C13 gradients, residual bound, finite losses, positive-preservation one-class behavior, mechanism residual non-collapse, and shortcut exclusion.
- Seed-0 validation-only gate: AUC at least C13 seed-0 `0.8655500226`, preferred gain `+0.005`, sensitivity at least `0.55`, specificity at least `0.75`, no material balanced-accuracy decrease, positive-negative gap decrease no more than `0.02`, inversions no worse than C13, mean positive residual at least `-0.02`, no more than `25%` of positive residuals below `-0.10`, nonzero residual variance, no saturation, and shortcut audit pass.
- No formal C17 run is permitted before the seed-0 decision artifact explicitly authorizes it. All runtime evidence must be generated under `/home/linruixin/chen/conda/envs/ma` on the server against `/data/csb/DMEA-HT/HT_2025.12_25`.

### Operational Rollout Instruction

- From the user's instruction on `2026-07-13` onward, do not repeat two-epoch smoke runs for subsequent phases. After the required server static/synthetic gate passes, launch the formal multi-seed training directly with seeds `[0, 42, 3407]`.
- This rollout preference does not interrupt the already-running C17 seed-0 pilot and does not retroactively change its evidence contract. The current C17 formal run remains conditional on the completed seed-0 validation-only decision artifact; once authorized, it uses the fixed multi-seed configuration without another smoke stage.

### C17 Seed-0 Pilot Completion And Formal Authorization

- The server-only C17 seed-0 pilot driver completed normally at `2026-07-13T18:48:00+08:00`. Both fixed routes produced validation predictions and passed the run health gate `13/13`; no test predictions were generated.
- DEMA-R selected epoch `12` with validation AUC `0.8682661838`, sensitivity `0.7446808511`, specificity `0.7659574468`, balanced accuracy `0.7553191489`, positive-negative gap `0.3269412524`, and residual standard deviation `0.2076434934`.
- DEMA-RP selected epoch `9` with validation AUC `0.8700769579`, sensitivity `0.7446808511`, specificity `0.7659574468`, balanced accuracy `0.7553191489`, positive-negative gap `0.3349866019`, and residual standard deviation `0.2098335194`.
- DEMA-RP preserved positive evidence: mean positive logit residual `+0.4735545389`, `FN -> TP = 10`, `TP -> FN = 0`, and pairwise inversions changed from `297` to `287` with `15` repaired and `5` introduced.
- Route decision: `PROMOTE_DEMA_C17_POSITIVE_PRESERVATION`. The preferred AUC gain target was not treated as a hard requirement; the route met the baseline and all safety checks. Formal multi-seed training is authorized with the fixed DEMA-RP loss and seeds `[0, 42, 3407]`; no additional smoke stage is permitted.

### C17 Formal Multi-Seed Completion And Decision

- Formal training ran on the server only in `/home/linruixin/chen/conda/envs/ma` against `/data/csb/DMEA-HT/HT_2025.12_25`, from `2026-07-13T19:01:56+08:00` to `2026-07-13T19:10:00+08:00`. The training worktree was `/home/linruixin/chen/project/DMEA-HT-c17-dema-residual-v4` at commit `fc3154a`; no second smoke stage was run.
- The fixed formal route was DEMA-RP with seeds `[0, 42, 3407]`. Validation-AUC checkpoint selection remained inside `train.py`; test was evaluated only after selection and was not used for any decision.
- Per-seed validation results were: seed `0`, best epoch `9`, AUC `0.8700769579`; seed `42`, best epoch `3`, AUC `0.8768673608`; seed `3407`, best epoch `6`, AUC `0.8619284744`.
- Aggregate validation AUC was `0.8696242644 +/- 0.0074797246`, with range `0.8619284744` to `0.8768673608`. This is `+0.0031688547` over the frozen C13 mean validation AUC `0.8664554097`; the preferred `0.90` target was not reached and is not claimed.
- Reporting-only test AUC was `0.8450491308 +/- 0.0034170713`. It did not override the validation result and was not used for route or checkpoint selection.
- Final validation sensitivity and specificity means were `0.8014184397` and `0.7872340426`; balanced accuracy mean was `0.7943262411`. Balanced accuracy did not materially decrease versus the corresponding base predictions in any seed.
- Positive-evidence preservation passed across all seeds. Mean positive logit residuals were `+0.4735545370`, `+0.0964033604`, and `+0.3309848715` for seeds `0`, `42`, and `3407`; no seed had a `TP -> FN` transition. The fraction of positive residuals below `-0.10` was at most `0.0638297872`.
- Pairwise inversion counts decreased for every seed: `297 -> 287` for seed `0`, `277 -> 272` for seed `42`, and `311 -> 305` for seed `3407`. The audit recorded `41` repaired and `20` introduced inversions across the three seeds.
- Residuals were nonzero and unsaturated for every seed. The largest shortcut-only label AUC across selected visit count, image usage/padding, bio availability/missingness, and report length was `0.5088275238`; the shortcut audit passed.
- The formal gate passed all required checks, including the exact seed contract, validation AUC above C13, AUC standard deviation at most `0.02`, sensitivity/specificity floors, balanced-accuracy safety, positive preservation, residual health, inversion non-worsening, inversion decrease in `3/3` seeds, shortcut safety, and test-as-reporting-only compliance.
- Final decision: `PROMOTE_DEMA_C17_POSITIVE_PRESERVATION`. The formal audit collector was committed as `eef2338` and executed in the isolated server worktree `/home/linruixin/chen/project/DMEA-HT-c17-formal-audit`; its gate and reports are under `analysis_reports/phase_c17_dema/` there.

## 2026-07-13 Single-Project Consolidation And C18 Preparation

### Consolidation Result

- Official model name remains `DEMA-HT`; historical repository and Python package identifiers remain `DMEA-HT` and `dmea_ht`.
- All server development work was consolidated into `/home/linruixin/chen/project/DMEA-HT` on the existing branch `feature/c16-disease-state-shared-specific-alignment`.
- The canonical server commit after code merge and the consolidation verifier is `759353acf128364a24ad0ae5ac3d2e822c1a0028`.
- C17 code, three validation-selected checkpoints, validation predictions, reporting-only test predictions, metrics, positive-preservation audit, inversion audit, shortcut audit, formal gate, and final report were migrated and reproduced before cleanup.
- Canonical C17 reproduction status: `CANONICAL_DMEA_HT_VERIFIED`; validation AUC mean remains `0.8696242644 +/- 0.0074797246`.
- Artifact migration preserved `1606` identical existing files, migrated `132` files, and retained `3` hash conflicts under `analysis_reports/project_consolidation/conflicts/` without overwrite.
- The `76` untracked files from the eight old worktrees were archived and SHA256-verified under `/home/linruixin/chen/project_archive/dema_ht_consolidation_20260713_205355/old_worktrees/` before cleanup.
- Removed registered worktrees: `DMEA-HT-c16`, `DMEA-HT-c16-mea`, `DMEA-HT-c16-mea-impl`, `DMEA-HT-c17-dema-residual`, `DMEA-HT-c17-dema-residual-v2`, `DMEA-HT-c17-dema-residual-v3`, `DMEA-HT-c17-dema-residual-v4`, and `DMEA-HT-c17-formal-audit`.
- The only remaining registered server worktree is the canonical `DMEA-HT`. `main` remains the repository baseline; the unmerged remote `origin/codex/c16-mea-design-audit` remains retained and documented rather than deleted.
- No new GitHub branch, Git worktree, or `DMEA-HT-*` server project directory was created. No reset, clean, force push, or destructive overwrite was used.
- The C17 initialization checkpoint path was updated from the removed historical worktree to the migrated canonical artifact path; this is a path-compatibility fix and does not alter C17 outputs.
- Consolidation status: `DMEA_HT_SINGLE_PROJECT_CONSOLIDATION_COMPLETE`.

### C18 Implementation Contract

- C18 freezes the C13 temporal-focus predictor and uses `base_logit` as the only base prediction input. Saved validation/test prediction CSVs are not training inputs.
- C18 reuses the existing DEMA evidence projectors, mechanism relation layer, conflict aggregator, and mechanism representation. No new encoder, modality, graph node, shared/private representation, or DecAlign module is introduced.
- Directional residuals are bounded support/opposition evidence corrections with separate support/opposition gates and deterministic `conflict_suppression = 1 - conflict_score`.
- C18-D uses BCE plus `0.001 * mean(effective_support_delta^2 + effective_opposition_delta^2)` plus `0.02 * positive_preservation`.
- C18-DH uses the identical objective plus `0.01 * hard_pair_ranking_loss` only for training-batch positive-negative pairs whose frozen-base margin is below `0.50`; single-class and no-pair cases are graph-connected zero.
- Both formal configs use seeds `[0, 42, 3407]`, validation AUC checkpoint selection, and reporting-only test evaluation. No C18 smoke or seed-0-only pilot is authorized or planned.
- Static and synthetic checks passed in `/home/linruixin/chen/conda/envs/ma` with `33/33 PASS` and authorized `DIRECT_MULTI_SEED_AUTHORIZED` before direct formal launch.

### C18 Direct Multi-Seed Completion And Decision

- C18 formal training ran server-only in `/home/linruixin/chen/conda/envs/ma` against `/data/csb/DMEA-HT/HT_2025.12_25` on `NVIDIA GeForce RTX 5090`, from `2026-07-13T21:40:39+08:00` to `2026-07-13T22:06:42+08:00`.
- The canonical run directories were `runs/dema_ht_c18_directional_multiseed/` and `runs/dema_ht_c18_directional_hardrank_multiseed/`. Both routes used the fixed seeds `[0, 42, 3407]`, validation AUC-only checkpoint selection, and reporting-only test evaluation.
- No C18 smoke run and no seed-0-only pilot were run. The formal driver ran C18-D followed sequentially by C18-DH after the static/synthetic gate.
- C18-D per-seed validation AUC was `0.8727931191`, `0.8850158443`, and `0.8646446356` for seeds `0`, `42`, and `3407`; mean/std was `0.8741511996 +/- 0.0102532835`.
- C18-DH per-seed validation AUC was `0.8750565867`, `0.8768673608`, and `0.8632865550` for seeds `0`, `42`, and `3407`; mean/std was `0.8717368342 +/- 0.0073739500`.
- Both C18 routes exceeded the C17 reference mean validation AUC `0.8696242644`, had validation-AUC standard deviation at most `0.02`, improved validation inversion count in all three seeds, and passed training validity, sensitivity, specificity, branch/gate health, shortcut, and test-reporting-only checks.
- C18-D reduced aggregate validation inversions from `885` to `834`, with `111` repaired and `60` introduced; C18-DH reduced them from `885` to `850`, with `81` repaired and `46` introduced. These reductions did not override the evidence-safety failures.
- Positive-preservation failed for both routes. C18-D seed `42` had mean positive directional delta `-0.0279701053` and `40.4255%` of positive deltas below `-0.10`; C18-DH seed `42` had mean positive directional delta `-0.0593720828` and the same `40.4255%` fraction. C18-D had `1` `TP -> FN`; C18-DH had `0` `TP -> FN`, but its positive residual suppression check still failed.
- Negative-inflation failed for both routes. The largest mean negative directional delta was `0.3605688866` for C18-D and `0.4102178146` for C18-DH; the largest negative probability increase was `0.0533094004` and `0.0667331754`, respectively.
- Directional branches remained finite and nonzero with unsaturated residual variation, and the maximum shortcut-only label AUC was `0.5088275238`. These are safety diagnostics only and do not rescue the failed evidence gates.
- Reporting-only test AUC means were `0.8433484505` for C18-D and `0.8450491308` for C18-DH. Test values were not used for checkpoint selection, route comparison, or promotion.
- The formal collector decision is `DEMA_C18_NEGATIVE_INFLATION`, with the additional failure label `DEMA_C18_POSITIVE_SUPPRESSION`. Selected route remains `C17`; current strict best remains `DEMA-HT C17 Positive Preservation`. C18 did not reach validation AUC `0.90` and neither route is promoted.
- C18 formal reports are under `analysis_reports/phase_c18_dema/`, including root-level merged route audits with a `route` column, route-specific audit subdirectories, metrics, transition analysis, gate JSON, and final report. The report-only collector layout fix was committed as `e3e85e1` and pulled by the canonical server before regeneration.
- Final C18 status: retain C17, do not alter the frozen C17 route based on C18 test or raw AUC, and do not start another C18 tuning run without a new evidence-gated plan.

## 2026-07-13 Phase C19 Polarity-Locked Residual

### Implementation Contract

- Official model name remains DEMA-HT; the canonical project is /home/linruixin/chen/project/DMEA-HT.
- The verified canonical history is on the existing main branch at merge-verification commit f97ee4d.
- No new GitHub branch, worktree, or DMEA-HT-c19 directory is created.
- C19 freezes the promoted C17 Positive Preservation checkpoint for each seed and does not read saved prediction CSVs as training inputs.
- C19-A is validation-only and audits support/opposition polarity, branch compensation, inversion transitions, and conflict behavior before model construction.
- MonotonicSupportCalibrator and MonotonicOppositionCalibrator have fixed positive slopes and do not cross-read the opposing evidence field.
- The residual sign is locked by tanh(q_support - q_opposition); uncertainty and conflict control magnitude only.
- EvidenceMagnitudeHead reads only absolute polarity, confidence, absolute frozen C17 logit, and valid mechanism norm. The correction magnitude is bounded by 0.20.
- The fixed loss is L_cls + 0.01*L_polarity + 0.02*L_positive + 0.02*L_negative + 0.001*L_magnitude.
- Validation AUC is the only checkpoint and decision metric. AUPRC is excluded from C19 reports, and test remains reporting-only.
- No smoke run and no seed-0-only pilot are permitted. After C19-A plus static/synthetic checks, formal seeds are [0, 42, 3407].
- Local Python verification was limited to syntax compilation and git diff --check; data/model runtime remains server-only.

### C19-A Gate Result

- Implementation commit 018f3b3 was pulled on server canonical main at /home/linruixin/chen/project/DMEA-HT.
- Server runtime environment: /home/linruixin/chen/conda/envs/ma; GPU: NVIDIA GeForce RTX 5090 with 32607 MiB.
- The C19-A audit was validation-only and read no test prediction files.
- All non-C19-A static, compile, checkpoint, legacy-config, and shortcut-exclusion checks passed (24/24).
- C19-A found nonconstant support-opposition gaps for all seeds, but the polarity direction was not stable:
  - seed 0: positive support-dominant 0.7446808511; negative opposition-dominant 0.3829787234;
  - seed 42: positive support-dominant 0.1063829787; negative opposition-dominant 0.6808510638;
  - seed 3407: positive support-dominant 0.2127659574; negative opposition-dominant 0.4680851064.
- The minimum 0.60/0.60 polarity admissibility condition therefore failed. The C19-A decision is C19_POLARITY_BASE_INVALID.
- Static/synthetic gate status is C19_DIRECT_MULTI_SEED_BLOCKED; synthetic checks were correctly skipped after C19-A failure.
- No C19 formal training process was started. No smoke, no seed-0 pilot, and no test/AUPRC-based decision occurred.
- Final action: retain frozen C17 Positive Preservation as the current strict best, with mean validation AUC 0.8696242644 +/- 0.0074797246.

### Operational Execution Rule

- When every explicitly defined static, synthetic, evidence, and safety gate passes, launch the authorized formal multi-seed training automatically with seeds [0, 42, 3407].
- Do not wait for an additional user confirmation after a passing gate.
- When any required gate fails, do not launch training or bypass the gate; record the failure and retain the current strict-best route.

## 2026-07-14 Phase C20 Mechanism Evidence Identifiability Audit

### C20 Contract And Implementation

- Official model and research name remains DEMA-HT. The canonical project remains `/home/linruixin/chen/project/DMEA-HT` on `main`; no new branch, worktree, or DMEA-HT-c20 directory was created.
- C19 remained blocked by `C19_POLARITY_BASE_INVALID`. C20 did not lower, bypass, or reinterpret that gate.
- C20 is analysis-only. It evaluates the three C17 validation-selected checkpoints with `eval()` and `torch.no_grad()`, without an optimizer, backward pass, new prediction module, training config, smoke run, seed-0 pilot, or formal training.
- C20 reads train and validation data only. Test data and test prediction files were not read. Saved C17 validation predictions were used only for patient/label/probability reproduction, never as training inputs.
- Cross-seed comparison uses patient-aligned linear CKA, patient-distance Spearman, kNN overlap, train-fit orthogonal Procrustes generalization, and scalar rank/sign consistency. Coordinate-wise cosine or raw coordinate correlation is not used as the identifiability criterion.
- Fixed diagnostic probes use train-fit standardization and L2 logistic regression with `C=1.0`; validation is evaluation-only. Random-label sanity uses fixed seed `20260714`. Shortcut fields and patient IDs are audit/alignment-only and excluded from all probes.
- Local implementation commits were `e6cdc14`, `85fe7f9`, `92cdb8e`, and `0ef6615`; all were pushed to `origin/main`. The server initially exported the NPZ at `92cdb8e`, then fast-forwarded to `0ef6615` for the final analysis. GitHub HTTPS pulls from the server timed out, so the already-pushed commits were transferred as a verified Git bundle and fast-forwarded without server-side source edits.

### C20 Runtime And Reproduction

- Runtime was server-only under `/home/linruixin/chen/conda/envs/ma` on `NVIDIA GeForce RTX 5090`, against `/data/csb/DMEA-HT/HT_2025.12_25` and manifest `manifest_distmatch_structmatch_evidence_v2_c13_temporal_focus.jsonl`.
- C17 checkpoint train/validation counts were `602/94` for each seed. Server-only NPZ artifacts are under `analysis_reports/phase_c20_dema/`; train SHA256 is `bede4e0b2ad4833942337b85782898299b839535dcb667bcfadef3a34e6ef062`, validation SHA256 is `d27ae718df583f7b80a1bf54169e4e2e19802bd347e474ad2352fd2a7cbc6edf`.
- C17 validation reproduction passed for all seeds: seed `0` max/mean absolute probability difference `1.11022302463e-16/2.08535907218e-17`; seed `42` `1.11022302463e-16/2.21823150864e-17`; seed `3407` `1.11022302463e-16/2.04475916105e-17`. Patient IDs and labels matched exactly.

### C20 Identifiability Results

- Stage means (linear CKA, distance Spearman, kNN Jaccard, fixed-probe validation AUC) were: raw modality encoders `0.9527, 0.9535, 0.6628, 0.6888`; evidence-role pooling `0.7924, 0.7828, 0.5815, 0.6743`; mechanism propagation `0.6618, 0.6483, 0.5307, 0.6967`; role scoring `0.4557, 0.5045, 0.2227, 0.8083`; mechanism aggregation `0.3951, 0.3490, 0.2415, 0.7479`; scalar compression `0.1675, 0.1023, 0.0760, 0.6563`.
- The first material instability is `mechanism_propagation`: its stage mean CKA `0.6618` is below `0.70` (stage distance Spearman is `0.6483`), while individual mechanism-layer/minimum checks also fail; mechanism morphology had CKA `0.6651`, and mechanism final representation had CKA `0.3871`, distance Spearman `0.4952`, and kNN Jaccard `0.3249`.
- The strongest internal probe was evidence-role logits/probabilities at mean validation AUC `0.8093`, with only one of three seeds at or above `0.83`; no internal candidate reached the required mean probe AUC `0.8396` and two-of-three seed condition. The final predictor probe reached `0.8696242644` but is not an intermediate mechanism candidate and cannot authorize C21.
- Hard-patient overlap did not improve over non-hard patients at the mechanism final layer: mean kNN Jaccard was approximately `0.3124` versus `0.3296`. C18-repaired patients numbered `12` and introduced patients `2`; their low mechanism-layer overlaps remain audit-only and do not support a new route.
- Shortcut audit found a strong raw `raw_n_visits` label association (orientation-invariant AUC about `0.9769`), but it was not used as a model feature, representation, or probe input. This remains a data shortcut warning, not evidence for C20 route promotion.

### C20 Gate And Decision

- Static gate, C17 export, reproduction, cross-seed analysis, probe, and transition analysis all completed successfully. No stable internal evidence/mechanism layer passed the simultaneous CKA, distance, kNN, probe, subgroup, and random-label conditions.
- C20 decision: `C20_INSTABILITY_FROM_MECHANISM_PROPAGATION`.
- C21 decision: `C21_NOT_AUTHORIZED`.
- Final action: `KEEP_DEMA_C17_STRICT_BEST` / `DEMA_C17_POSITIVE_PRESERVATION`, with mean validation AUC `0.8696242644 +/- 0.0074797246`. Do not start C21 training, residual polarity expansion, or scalar-compression rescue from this C20 result without a new evidence-gated plan.
- Reports: `analysis_reports/phase_c20_dema/c20_route_decision.md` and `analysis_reports/phase_c20_dema/phase_c20_dema_final_report.md`.

## 2026-07-14 Phase C21-A Mechanism Propagation Responsibility Audit

### Contract And Scope

- Official project and model name remain DEMA-HT. The canonical server worktree is `/home/linruixin/chen/project/DMEA-HT` on `main`, with data root `/data/csb/DMEA-HT/HT_2025.12_25` and runtime `/home/linruixin/chen/conda/envs/ma`.
- C21-A is analysis-only. It does not train, optimize, backpropagate, alter labels, alter patient-level splits, alter manifests, alter the task definition, read test data, calculate AUPRC, create a branch/worktree, or authorize a new model.
- The audit uses the frozen C17 checkpoints for seeds `[0, 42, 3407]`, validation AUC as the primary metric, and train/validation data only. Test remains reporting-only and was not read.
- The graph is taken from the actual `HTMechanismRelationLayer`: image/text morphology to M1, bio immune/function to M2/M3, text opposition/temporal to M4/M5, five mechanism states into final `MultiheadAttention`, and text-global/bio-other additive context edges. Independent relation edge weights and explicit residual node updates are recorded as unavailable rather than invented.

### Implementation And Server Verification

- C21-A implementation commits were `82ab4b0`, `dd50265`, `48fbdce`, and `5132ba1`; all were pushed to GitHub `main` and the server canonical worktree was fast-forwarded to `5132ba1`.
- Added the read-only trace exporter, node/edge stability analyses, edge ablations, node/modulation bypasses, responsibility scorer, and final collector under `scripts/*phase_c21a*`.
- The first mirror reproduction attempt exposed two numerical-path issues, both corrected without loosening thresholds: exact context addition order and preserving the real attention key/value identity when no attention edge is intervened on.
- The final server static gate and C17 reproduction gate passed. For every seed, validation patient IDs and labels matched exactly; maximum absolute probability difference was `1.1102230246251565e-16` and mean difference was approximately `2.04e-17` to `2.22e-17`.
- Server C17 validation counts were `602/94` for train/validation per seed. Large trace NPZ archives remain server-only. The tensor inventory records complete split shapes and marks unavailable graph fields explicitly.

### C21-A Results

- The frozen C17 baseline validation AUCs remained `0.8700769579`, `0.8768673608`, and `0.8619284744`; mean/std remains `0.8696242644 +/- 0.0074797246`.
- The highest supported responsibility candidate was `M3_function_to_final_mechanism`, with responsibility score `0.8001348`, mean absolute probability effect `0.0002216`, and strong within-seed propagation deformation. Its cross-seed ablation Spearman was only `0.0078386`, despite sign consistency `1.0`; this is unstable magnitude responsibility, not reproducible localization.
- The next supported candidate was `text_morphology_to_M1_morphology` with score `0.2377121`, cross-seed ablation Spearman `0.3143376`, and direction consistency `0.5673759`, also below the localization gate. Role scoring and aggregation had large cross-seed instability but no direct supported ablation intervention.
- Validation-only shortcut audit found orientation-invariant AUC `0.9769126` for `raw_n_visits` and `0.9418289` for `raw_n_images`; selected visit count and used image count were `0.5`. These fields were excluded from the model and all probes.

### Decision

- C21-A route: `C21A_DIFFUSE_MECHANISM_PROPAGATION_INSTABILITY`.
- `localized_reproducible = False`; `C22_DESIGN_AUTHORIZED = False`; `training_authorized = False`.
- Retain `DEMA_C17_POSITIVE_PRESERVATION` as the strict-best route. Do not start C22 design or any new training from this audit result.
- Final server report and artifacts are under `analysis_reports/phase_c21a_dema/`, including `phase_c21a_dema_final_report.md`, reproduction checks, tensor inventory, node/edge stability tables, ablation summaries, responsibility scores, shortcut exclusion audit, and command log.

## 2026-07-14 Phase C22 Stable Evidence Pooling Direct Multi-Seed

### Pre-Edit Contract

- Official project and research name remain `DEMA-HT`; repository and Python package identifiers remain `DMEA-HT` and `dmea_ht`.
- C21-A found diffuse, non-localizable mechanism-propagation instability. The user-authorized C22 experiment is a whole-stage bypass falsification, not a claim that any individual clinical mechanism node has been identified.
- The canonical server worktree remains `/home/linruixin/chen/project/DMEA-HT` on `main`; the authoritative data root is `/data/csb/DMEA-HT/HT_2025.12_25`; runtime is `/home/linruixin/chen/conda/envs/ma`.
- No branch, worktree, project copy, smoke run, or seed-0 pilot is permitted. After the static/synthetic gate passes, launch formal seeds `[0, 42, 3407]` directly.
- C13 labels, patient-level splits, history cutoff, manifest, task definition, and validation-AUC checkpoint selection remain frozen. Test remains reporting-only.
- C22 freezes the C13 base predictor and the C17 evidence projector state, pools only the 14 real pre-propagation image/text/bio projector nodes by valid-mask mean, and bypasses mechanism propagation, downstream role scoring, conflict aggregation, and the final mechanism head.
- Only the new stable-evidence residual head is trainable. It uses hidden size `256`, the C17 activation/dropout pattern, zero-initialized output, `delta = 0.50 * tanh(raw_delta)`, and `final_logit = base_logit + delta`.
- The fixed loss is `BCEWithLogits + 0.001 * mean(delta^2) + 0.02 * positive_preserve`, where `positive_preserve = mean_positive(relu(-delta - 0.05))`; the all-negative case remains graph-connected zero.
- Validation AUC is the sole selection, route-comparison, promotion, and rejection metric. No AUPRC field is generated for C22 formal reports. Sensitivity, specificity, balanced accuracy, positive preservation, inversion counts, residual health, and shortcut audits are safety diagnostics only.
- Promotion requires C22 mean validation AUC above C17, at least two of three seeds above C17, no seed drop greater than `0.005`, standard deviation at most `0.02`, positive-preservation and inversion guards, residual health, and selected-structure shortcut-only AUC at most `0.55`.

### Implementation

- Added `dmea_ht/c22_stable_pooling.py`, `configs/dema_ht_c22_stable_evidence_pooling_multiseed.yaml`, `scripts/gate_phase_c22_stable_pooling.py`, `scripts/train_phase_c22.py`, and `scripts/collect_phase_c22_formal_report.py`.
- The C22 training entry point is isolated from `train.py`; existing C17, C18, and C19 training branches are unchanged.
- Required formal artifacts are written under `runs/dema_ht_c22_stable_evidence_pooling_multiseed/` and `analysis_reports/phase_c22_dema/`, including epoch/seed/summary metrics, patient diagnostics, positive-preservation audit, pairwise ranking/inversion tables, residual health, shortcut residual audit, C13/C17 comparison, necessity report, seed-stability report, and final decision report.
- Local verification is limited to syntax/static/synthetic checks; no data or model training is run on the local machine.

### Runtime And Final Decision

- The first two launch attempts stopped before training because the C22 config contained two historical path spelling errors: `dema_ht_v2_c13...` was corrected to the canonical `dmea_ht_v2_c13...`, and `dema_ht_c16...` was corrected to `dmea_ht_v2_c16...`. No training data, labels, split, or model code was changed for these path-only repairs.
- The server static/synthetic gate passed `12/12` under `/home/linruixin/chen/conda/envs/ma`. No smoke or seed-0 pilot was run after the gate.
- Formal C22 training ran server-only on `NVIDIA GeForce RTX 5090` from `2026-07-14T03:53:42+08:00` to `2026-07-14T04:05:36+08:00`, against `/data/csb/DMEA-HT/HT_2025.12_25`, with seeds `[0, 42, 3407]`. The training implementation commit was `dc6eba1`; the final audit collector fix was `bdb718f`.
- Validation AUC by seed was `0.8700769579` (seed 0), `0.8755092802` (seed 42), and `0.8578542327` (seed 3407). C22 mean/std was `0.8678134903 +/- 0.0090425461`, compared with C17 `0.8696242644 +/- 0.0074797246` and C13 `0.8664554097 +/- 0.0077356304`. All three C22 seeds were below the corresponding C17 AUC; the largest seed drop was `-0.0040742417`.
- Reporting-only test AUC was `0.8429705215 +/- 0.0020439633`. Test was evaluated only after validation-AUC checkpoint selection and was not used for any route decision.
- Positive preservation failed. Mean positive residuals were `+0.4738359217`, `-0.1796204704`, and `+0.2554593719` for seeds `0`, `42`, and `3407`. Seed 42 had `87.23404255%` of positive residuals below `-0.10`, with `2` C13 `TP -> FN` transitions and `4` C17 `TP -> FN` transitions.
- Pairwise inversion non-worsening failed: C17 to C22 inversion counts were `287 -> 287`, `272 -> 275`, and `305 -> 314` for seeds `0`, `42`, and `3407`. The corrected audit recorded repaired/introduced pairs of `10/10`, `6/9`, and `2/11` respectively.
- Residual health passed: all seeds had nonzero residual variance and no bound saturation. The selected-structure shortcut-only label AUC was `0.4762084402`, below the `0.55` audit threshold; shortcut fields remained audit-only.
- Final C22 decision: `DEMA_C22_POSITIVE_SUPPRESSION`. C22 is not promoted. The strict-best route remains `DEMA_C17_POSITIVE_PRESERVATION`; C22 does not support a claim that a particular mechanism node is clinically necessary.
- Final server artifacts are under `analysis_reports/phase_c22_dema/` and the formal run is under `runs/dema_ht_c22_stable_evidence_pooling_multiseed/`.

## 2026-07-14 Phase C23 Confidence-Gated Local Residual Direct Multi-Seed

### Pre-Edit Contract

- Official project and research name remain `DEMA-HT`; the canonical server worktree remains `/home/linruixin/chen/project/DMEA-HT` on `main`. No branch, worktree, project copy, smoke run, or seed-0 pilot is permitted.
- The authoritative data root is `/data/csb/DMEA-HT/HT_2025.12_25`, the frozen manifest is `manifest_distmatch_structmatch_evidence_v2_c13_temporal_focus.jsonl`, and the runtime is `/home/linruixin/chen/conda/envs/ma`.
- C23 freezes the complete validation-selected C17 route for each seed, including C13, encoders, evidence projectors, mechanism propagation, role scoring, conflict aggregation, mechanism head, and the C17 residual head.
- The C23 base logit is the full frozen C17 final logit. The new head reads only the frozen C17 `mea_mechanism_state`, described conservatively as a latent pathological-mechanism interaction representation rather than an identified clinical node.
- The deterministic non-learnable gate is `exp(-abs(frozen_c17_logit) / 1.0)`. The bounded correction is `0.15 * gate.detach() * tanh(raw_delta_c23)`, and only the new C23 residual head is trainable.
- The fixed objective is `BCEWithLogits + 0.001 * mean(delta^2) + 0.02 * positive_preserve + 0.02 * negative_preserve + 0.01 * high_confidence_preserve`. Missing-mask terms remain graph-connected zero.
- Validation AUC is the sole checkpoint-selection and route-decision metric. Test remains reporting-only after validation selection. Shortcut fields remain audit-only and are excluded from the model and loss.
- Once the full server path/reproduction gate prints `C23_DIRECT_MULTI_SEED_AUTHORIZED`, formal seeds `[0, 42, 3407]` must launch directly without further confirmation.

### Implementation And Local Verification

- Added `dmea_ht/c23_confidence_gated_residual.py`, `configs/dema_ht_c23_confidence_gated_residual_multiseed.yaml`, `scripts/gate_phase_c23_confidence_gated_residual.py`, `scripts/train_phase_c23.py`, and `scripts/collect_phase_c23_formal_report.py`.
- The independent training path records per-epoch loss components, validation AUC and threshold diagnostics, gate statistics, positive/negative residual behavior, confidence-group residual magnitude, bound proximity, and pairwise inversion counts.
- The collector produces the required patient diagnostics, C17 transitions, positive/negative preservation audit, confidence-group and high-confidence damage audits, pairwise ranking/inversion tables, residual-health and shortcut audits, C17 comparison, seed-stability report, and final decision report.
- Local syntax checks passed. The local static/synthetic gate passed all `23/23` checks, including initial C17/C23 logit equality, deterministic gate range and monotonicity, residual bound, finite nonzero new-head gradient, graph-connected one-class zeros, shortcut exclusion, prediction-file exclusion, and legacy-config parsing.
- Local execution was limited to syntax and synthetic tensors. No local data loading, checkpoint inference, or training was performed.

### Server Runtime And Final Decision

- Implementation commit: `4777bef` (`Implement C23 confidence-gated residual`).
- Server gate commit and resolved path inventory: pending.
- Training start time, GPU, completion time, per-seed validation AUC, preservation gates, inversion audit, shortcut audit, and final decision: pending formal server execution.
