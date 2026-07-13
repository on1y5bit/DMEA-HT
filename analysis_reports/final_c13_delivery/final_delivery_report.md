# DMEA-HT Final C13 Delivery Report

## Final Decision

`FREEZE_C13_AS_STRICT_BEST_AND_REPORT_LIMITATION`

Delivery verification: `PASS`.

## Frozen Performance

| seed | best_epoch_metrics_csv | val_auc | val_auprc | val_sensitivity | val_specificity | test_auc_reporting_only |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 16 | 0.8655500226346763 | 0.8614940382510322 | 0.5319148936170213 | 0.8936170212765957 | 0.854875283446712 |
| 42 | 17 | 0.8746038931643277 | 0.8578789281513747 | 0.8297872340425532 | 0.8085106382978723 | 0.8418367346938775 |
| 3407 | 15 | 0.8592123132639203 | 0.8517620303864883 | 0.5957446808510638 | 0.851063829787234 | 0.8412698412698412 |

Three-seed validation AUC is `0.8665 +/- 0.0077`. Three-seed validation AUPRC is `0.8570 +/- 0.0049`.

## Shortcut Safety

| split | selection_role | max_abs_spearman | linear_r2_prob_from_shortcuts | shortcut_only_label_auc_audit_only |
| --- | --- | --- | --- | --- |
| val | selection_safety | 0.1548552043965999 | 0.0600662470693158 | 0.4762084402193048 |
| test | reporting_only | 0.1745086883635883 | 0.0587189073524456 | 0.3982741244646006 |

## Hard-Subgroup And Data Limitation

| phase | route | status | total_pairwise_rows | total_inversion_rows | all_seed_inversion_pairs | hard_positive_count | hard_negative_count | positive_matching_coverage | negative_matching_coverage | training_authorized | decision_basis |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C14-C | HARD_PATIENT_SUBGROUP_FAILURE | C14C_HARD_SUBGROUP_STOP | 6627 | 885 | 215 |  |  |  |  | 0 | {"fusion_interaction_fraction": 0.08553459119496855, "fusion_margin_means": [-0.7262771141070586, -1.0607832556997892, -0.8567424396246565], "fusion_margin_negative_seed_count": 3, "image_margin_means": [-0.1641098655617008, -0.11839184983546147, -0.12391641076427207], "image_margin_negative_seed_count": 3, "image_opposed_fraction": 0.28553459119496855, "image_repair_means": [-0.7960348729903881, -1.0353188159060664, -0.901845560065307], "image_repair_positive_seed_count": 0, "majority_inversion_pairs": 75, "stable_inversion_pairs": 215, "text_driven_fraction": 0.6289308176100629, "text_margin_means": [-0.035670436381434016, -0.005998677930620033, -0.03332760452716795], "text_margin_nonpositive_seed_count": 3, "top5_patient_inversion_share": 0.5932203389830508} |
| C14-E | DATA_LIMIT_NO_GENERAL_MODEL_FIX | KEEP_C13_AND_REPORT_LIMITATION |  |  |  | 36 | 43 | 0.3055555555555556 | 0.0930232558139534 | 0 | Matched-control coverage is below 50% for at least one label subgroup, preventing a broad model mechanism claim. |

C14-E did not identify a broad, matched-control-supported mechanism. The final action is to keep C13 and report the data limitation, not to launch C15.

## Artifact Verification

| check | status | evidence |
| --- | --- | --- |
| formal_seeds | PASS | [0, 42, 3407] |
| primary_metric | PASS | val_AUC |
| manifest_path | PASS | /data/csb/DMEA-HT/HT_2025.12_25/manifest_distmatch_structmatch_evidence_v2_c13_temporal_focus.jsonl |
| manifest_rows | PASS | 780 |
| manifest_split_label_counts | PASS | {"test_0": 42, "test_1": 42, "train_0": 301, "train_1": 301, "val_0": 47, "val_1": 47} |
| checkpoints_exist | PASS | [1, 1, 1] |
| checkpoints_load | PASS | ['loaded', 'loaded', 'loaded'] |
| checkpoint_seed_metadata | PASS | [{'seed': 0, 'checkpoint_seed': 0}, {'seed': 42, 'checkpoint_seed': 42}, {'seed': 3407, 'checkpoint_seed': 3407}] |
| checkpoint_epoch_metadata | PASS | [{'seed': 0, 'best_epoch_checkpoint': 16, 'best_epoch_metrics_csv': 16}, {'seed': 42, 'best_epoch_checkpoint': 17, 'best_epoch_metrics_csv': 17}, {'seed': 3407, 'best_epoch_checkpoint': 15, 'best_epoch_metrics_csv': 15}] |
| validation_prediction_rows | PASS | [94, 94, 94] |
| test_prediction_rows_reporting_only | PASS | [84, 84, 84] |
| c14b_reproduction | PASS | [{'seed': 0, 'max_abs_prob_diff': 1.1102230246251563e-16, 'reproduction_pass': 1}, {'seed': 42, 'max_abs_prob_diff': 1.1102230246251563e-16, 'reproduction_pass': 1}, {'seed': 3407, 'max_abs_prob_diff': 1.1102230246251563e-16, 'reproduction_pass': 1}] |
| validation_auc_frozen | PASS | 0.8664554096876415 |
| shortcut_safety_recorded | PASS | {'max_abs_spearman': 0.1548552043965999, 'linear_r2_prob_from_shortcuts': 0.0600662470693158, 'shortcut_only_label_auc_audit_only': 0.4762084402193048} |
| c14e_training_blocked | PASS | {'route': 'DATA_LIMIT_NO_GENERAL_MODEL_FIX', 'allowed_next_step': 'KEEP_C13_AND_REPORT_LIMITATION', 'training_authorized': 0} |
| inventory_complete | PASS | [] |

## Selection Integrity

- Patient-level labels, split assignment, task definition, C13 manifest, images, bio values, and report construction are frozen.
- Checkpoints were selected by validation AUC only.
- Test metrics are reporting-only.
- No shortcut field is a classifier input.
- No C15 or post-C14-E training was authorized.

The validation AUC 0.90 target was not reached. C13 remains the final strict best under the available evidence.
