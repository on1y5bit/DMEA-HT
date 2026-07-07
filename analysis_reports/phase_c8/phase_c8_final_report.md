# Phase C8 Final Report

## Route Status

- Current main path: strict structural matched DMEA-MVP.
- C1/C2/C6 remain ablation-only and are not revived in Phase C8.
- No training was performed in C8.
- Phase C7 report was loaded.

## Validation-Only Strict MVP Performance Recap

- Validation AUC: 0.7443.
- Validation AUPRC: 0.7192.
- Sensitivity/specificity at threshold 0.5: 0.8865 / 0.4326.
- Positive-negative prediction gap: 0.1797.
- Validation false negatives / false positives: 16 / 80.

## Error Taxonomy Summary

| split | error_type | n_errors | proportion_of_errors | false_negative_count | false_positive_count | mean_pred_prob | mean_abs_error |
| --- | --- | --- | --- | --- | --- | --- | --- |
| val | long_report_or_multivisit_uncertainty | 25 | 0.2604 | 0 | 25 | 0.6754 | 0.6754 |
| val | other_error | 25 | 0.2604 | 1 | 24 | 0.6786 | 0.6882 |
| val | morphology_positive_false_negative | 15 | 0.1562 | 15 | 0 | 0.3884 | 0.6116 |
| val | high_confidence_false_positive | 12 | 0.1250 | 0 | 12 | 0.8639 | 0.8639 |
| val | morphology_low_confidence_false_positive | 10 | 0.1042 | 0 | 10 | 0.6412 | 0.6412 |
| val | borderline_error | 9 | 0.0938 | 0 | 9 | 0.5466 | 0.5466 |

### Top False-Negative Categories

| split | error_type | n_errors | proportion_of_errors | false_negative_count | false_positive_count | mean_pred_prob | mean_abs_error |
| --- | --- | --- | --- | --- | --- | --- | --- |
| val | morphology_positive_false_negative | 15 | 0.1562 | 15 | 0 | 0.3884 | 0.6116 |
| val | other_error | 25 | 0.2604 | 1 | 24 | 0.6786 | 0.6882 |

### Top False-Positive Categories

| split | error_type | n_errors | proportion_of_errors | false_negative_count | false_positive_count | mean_pred_prob | mean_abs_error |
| --- | --- | --- | --- | --- | --- | --- | --- |
| val | long_report_or_multivisit_uncertainty | 25 | 0.2604 | 0 | 25 | 0.6754 | 0.6754 |
| val | other_error | 25 | 0.2604 | 1 | 24 | 0.6786 | 0.6882 |
| val | high_confidence_false_positive | 12 | 0.1250 | 0 | 12 | 0.8639 | 0.8639 |
| val | morphology_low_confidence_false_positive | 10 | 0.1042 | 0 | 10 | 0.6412 | 0.6412 |
| val | borderline_error | 9 | 0.0938 | 0 | 9 | 0.5466 | 0.5466 |

## Evidence Strata Findings

### Morphology Evidence Label

| stratum_name | stratum_value | n | auc_if_defined | auprc_if_defined | sensitivity_at_0p5 | specificity_at_0p5 | positive_negative_gap | false_negative_rate | false_positive_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| txt_morphology_label | 1 | 252 | 0.7331 | 0.7361 | 0.8889 | 0.4017 | 0.1660 | 0.1111 | 0.5983 |
| txt_morphology_label | 0 | 30 | 0.7083 | 0.3359 | 0.8333 | 0.5833 | 0.1466 | 0.1667 | 0.4167 |

### Morphology Evidence Confidence

| stratum_name | stratum_value | n | auc_if_defined | auprc_if_defined | sensitivity_at_0p5 | specificity_at_0p5 | positive_negative_gap | false_negative_rate | false_positive_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| txt_morphology_confidence_bin | high | 252 | 0.7331 | 0.7361 | 0.8889 | 0.4017 | 0.1660 | 0.1111 | 0.5983 |
| txt_morphology_confidence_bin | low | 30 | 0.7083 | 0.3359 | 0.8333 | 0.5833 | 0.1466 | 0.1667 | 0.4167 |

### Negative Evidence Label

| stratum_name | stratum_value | n | auc_if_defined | auprc_if_defined | sensitivity_at_0p5 | specificity_at_0p5 | positive_negative_gap | false_negative_rate | false_positive_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| txt_negative_label | 0 | 180 | 0.7690 | 0.7833 | 0.8222 | 0.5222 | 0.1872 | 0.1778 | 0.4778 |
| txt_negative_label | 1 | 102 | 0.7247 | 0.6892 | 1.0000 | 0.2745 | 0.1664 | 0.0000 | 0.7255 |

### Negative Evidence Confidence

| stratum_name | stratum_value | n | auc_if_defined | auprc_if_defined | sensitivity_at_0p5 | specificity_at_0p5 | positive_negative_gap | false_negative_rate | false_positive_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| txt_negative_confidence_bin | low | 123 | 0.8210 | 0.8560 | 0.8841 | 0.5185 | 0.2232 | 0.1159 | 0.4815 |
| txt_negative_confidence_bin | high | 102 | 0.7247 | 0.6892 | 1.0000 | 0.2745 | 0.1664 | 0.0000 | 0.7255 |
| txt_negative_confidence_bin | medium | 57 | 0.5767 | 0.4189 | 0.6190 | 0.5278 | 0.0518 | 0.3810 | 0.4722 |

### Report, Visit/Image, And Bio Availability Strata

| stratum_name | stratum_value | n | auc_if_defined | auprc_if_defined | sensitivity_at_0p5 | specificity_at_0p5 | positive_negative_gap | false_negative_rate | false_positive_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| report_length_bin | q1_low | 72 | 0.8563 | 0.8615 | 0.9487 | 0.5152 | 0.2680 | 0.0513 | 0.4848 |
| report_length_bin | q3_midhigh | 72 | 0.7374 | 0.7205 | 0.8974 | 0.3939 | 0.1838 | 0.1026 | 0.6061 |
| report_length_bin | q2_midlow | 69 | 0.7496 | 0.7206 | 0.9000 | 0.4615 | 0.1617 | 0.1000 | 0.5385 |
| report_length_bin | q4_high | 69 | 0.6094 | 0.6048 | 0.7879 | 0.3611 | 0.0918 | 0.2121 | 0.6389 |
| selected_n_visits_bin | high | 186 | 0.6809 | 0.6637 | 0.8817 | 0.3441 | 0.1363 | 0.1183 | 0.6559 |
| selected_n_visits_bin | low | 96 | 0.8581 | 0.8554 | 0.8958 | 0.6042 | 0.2637 | 0.1042 | 0.3958 |
| used_images_bin | high | 186 | 0.6809 | 0.6637 | 0.8817 | 0.3441 | 0.1363 | 0.1183 | 0.6559 |
| used_images_bin | low | 96 | 0.8581 | 0.8554 | 0.8958 | 0.6042 | 0.2637 | 0.1042 | 0.3958 |
| has_bio | 1 | 282 | 0.7443 | 0.7192 | 0.8865 | 0.4326 | 0.1797 | 0.1135 | 0.5674 |
| bio_missing_count_bin | 2plus | 264 | 0.7569 | 0.7586 | 0.8788 | 0.4470 | 0.1871 | 0.1212 | 0.5530 |
| bio_missing_count_bin | 0 | 18 | 0.5679 | 0.5973 | 1.0000 | 0.2222 | 0.0716 | 0.0000 | 0.7778 |

## High-Confidence Error Examples

| patient_id | label | pred_prob | error_type | matched_morphology_terms | report_length | selected_n_visits |
| --- | --- | --- | --- | --- | --- | --- |
| 10106168 | 0 | 0.9266 | high_confidence_false_positive | ['回声欠均', '低回声'] | 441 | 2 |
| 10106168 | 0 | 0.9266 | high_confidence_false_positive | ['回声欠均', '低回声'] | 441 | 2 |
| 10106168 | 0 | 0.9165 | high_confidence_false_positive | ['回声欠均', '低回声'] | 441 | 2 |
| 10084278 | 0 | 0.8726 | high_confidence_false_positive | ['回声不均', '回声欠均', '回声欠均匀', '低回声'] | 1739 | 5 |
| 10084278 | 0 | 0.8707 | high_confidence_false_positive | ['回声不均', '回声欠均', '回声欠均匀', '低回声'] | 1739 | 5 |
| 10027380 | 0 | 0.8699 | high_confidence_false_positive | ['实质回声不均', '回声不均', '回声欠均', '回声欠均匀', '低回声'] | 805 | 3 |
| 10009149 | 0 | 0.8634 | high_confidence_false_positive | ['实质回声不均', '回声不均', '低回声'] | 103 | 1 |
| 10084278 | 0 | 0.8625 | high_confidence_false_positive | ['回声不均', '回声欠均', '回声欠均匀', '低回声'] | 1739 | 5 |
| 10023011 | 0 | 0.8565 | morphology_low_confidence_false_positive | [] | 835 | 4 |
| 10034355 | 0 | 0.8256 | high_confidence_false_positive | ['低回声'] | 700 | 3 |
| 10038703 | 0 | 0.8169 | high_confidence_false_positive | ['回声欠均', '回声欠均匀', '低回声'] | 908 | 3 |
| 10043013 | 0 | 0.8124 | high_confidence_false_positive | ['实质回声不均', '回声不均', '回声欠均', '回声欠均匀', '低回声'] | 374 | 3 |

## Shortcut Audit Interpretation

Validation errors do not show a large audit-bin concentration by the configured threshold. Shortcut fields remain audit-only.

| field | bin | n | error_rate | fn_rate | fp_rate | mean_pred_prob | mean_label | auc_if_defined | error_rate_delta_vs_overall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| report_length | q4_high | 69 | 0.4348 | 0.2121 | 0.6389 | 0.6000 | 0.4783 | 0.6094 | 0.0944 |
| bio_missing_count | 0 | 18 | 0.3889 | 0.0000 | 0.7778 | 0.7385 | 0.5000 | 0.5679 | 0.0485 |
| selected_n_visits | high | 186 | 0.3871 | 0.1183 | 0.6559 | 0.6300 | 0.5000 | 0.6809 | 0.0467 |
| used_images | high | 186 | 0.3871 | 0.1183 | 0.6559 | 0.6300 | 0.5000 | 0.6809 | 0.0467 |
| report_length | q2_midlow | 69 | 0.3478 | 0.1000 | 0.5385 | 0.5942 | 0.4348 | 0.7496 | 0.0074 |
| image_padding_count | high | 282 | 0.3404 | 0.1135 | 0.5674 | 0.6194 | 0.5000 | 0.7443 | 0.0000 |
| has_bio | 1 | 282 | 0.3404 | 0.1135 | 0.5674 | 0.6194 | 0.5000 | 0.7443 | 0.0000 |
| bio_missing_count | 2 | 264 | 0.3371 | 0.1212 | 0.5530 | 0.6113 | 0.5000 | 0.7569 | -0.0033 |
| report_length | q3_midhigh | 72 | 0.3333 | 0.1026 | 0.6061 | 0.6513 | 0.5417 | 0.7374 | -0.0071 |
| selected_n_visits | low | 96 | 0.2500 | 0.1042 | 0.3958 | 0.5988 | 0.5000 | 0.8581 | -0.0904 |
| used_images | low | 96 | 0.2500 | 0.1042 | 0.3958 | 0.5988 | 0.5000 | 0.8581 | -0.0904 |
| report_length | q1_low | 72 | 0.2500 | 0.0513 | 0.4848 | 0.6303 | 0.5417 | 0.8563 | -0.0904 |

## Next-Phase Recommendation

`RETURN_TO_DATA_AUDIT`.

This recommendation is validation-based. Test outputs are reporting-only and did not drive the recommendation.

## Suggested Future Gate

- A future pilot may be considered only if the validation-set error pattern is concrete and reproducible.
- No test tuning.
- No shortcut variables as classifier inputs.
- Bad-seed or stress-seed pilot before formal training.
- Positive-preservation check before formal training.
- Formal seeds remain 0, 42, and 3407 only after the pilot gate passes.

## Inputs Used

| path | status | notes |
| --- | --- | --- |
| analysis_reports/phase_c8/strict_mvp_error_taxonomy_summary.csv | loaded | 12 rows |
| analysis_reports/phase_c8/strict_mvp_overall_metrics.csv | loaded | 2 rows |
| analysis_reports/phase_c8/strict_mvp_evidence_strata_val.csv | loaded | 22 rows |
| analysis_reports/phase_c8/strict_mvp_high_confidence_errors_val.csv | loaded | 13 rows |
| analysis_reports/phase_c8/strict_mvp_shortcut_strata_val.csv | loaded | 12 rows |
| analysis_reports/phase_c7/phase_c7_final_report.md | loaded | 8201 chars |
| analysis_reports/phase_c8/inputs_used_and_missing.csv | loaded | 7 rows |
| /data/csb/DMEA-HT/HT_2025.12_25/manifest_distmatch_structmatch_evidence_v2.jsonl | loaded | 780 manifest rows |
| /home/linruixin/chen/project/DMEA-HT/runs/dmea_ht_distmatch_structmatch_auc_20260703_145921/predictions/val_predictions_seed_0.csv | loaded | 94 rows |
| /home/linruixin/chen/project/DMEA-HT/runs/dmea_ht_distmatch_structmatch_auc_20260703_145921/predictions/val_predictions_seed_3407.csv | loaded | 94 rows |
| /home/linruixin/chen/project/DMEA-HT/runs/dmea_ht_distmatch_structmatch_auc_20260703_145921/predictions/val_predictions_seed_42.csv | loaded | 94 rows |
| /home/linruixin/chen/project/DMEA-HT/runs/dmea_ht_distmatch_structmatch_auc_20260703_145921/predictions/test_predictions_seed_0.csv | loaded | 84 rows |
| /home/linruixin/chen/project/DMEA-HT/runs/dmea_ht_distmatch_structmatch_auc_20260703_145921/predictions/test_predictions_seed_3407.csv | loaded | 84 rows |
| /home/linruixin/chen/project/DMEA-HT/runs/dmea_ht_distmatch_structmatch_auc_20260703_145921/predictions/test_predictions_seed_42.csv | loaded | 84 rows |
