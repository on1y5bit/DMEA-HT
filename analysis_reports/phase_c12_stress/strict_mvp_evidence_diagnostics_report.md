# Strict MVP Evidence Diagnostics Report

Target model: strict structural matched DMEA-MVP. Evidence fields and shortcut fields are analysis-only.

## Undefined Validation AUC/AUPRC Strata

_No rows._

## Validation Strata With Highest False-Negative Rates

| stratum_name | stratum_value | n | false_negative_rate | mean_pred_prob_positive |
| --- | --- | --- | --- | --- |
| txt_negative_confidence_bin | medium | 57 | 0.6190 | 0.4867 |
| report_length_bin | q4_high | 69 | 0.3636 | 0.5722 |
| txt_morphology_label | 0 | 36 | 0.3333 | 0.5815 |
| txt_morphology_confidence_bin | low | 36 | 0.3333 | 0.5815 |
| matched_morphology_terms_present | absent | 36 | 0.3333 | 0.5815 |
| txt_negative_label | 0 | 180 | 0.2889 | 0.6235 |
| selected_n_visits_bin | high | 186 | 0.2581 | 0.6399 |
| used_images_bin | high | 186 | 0.2581 | 0.6399 |
| report_length_bin | q2_midlow | 69 | 0.2333 | 0.6694 |
| txt_morphology_confidence_bin | medium | 132 | 0.2292 | 0.6634 |

## Validation Strata With Highest False-Positive Rates

| stratum_name | stratum_value | n | false_positive_rate | mean_pred_prob_negative |
| --- | --- | --- | --- | --- |
| txt_morphology_confidence_bin | high | 114 | 0.7778 | 0.6148 |
| bio_missing_count_bin | 0 | 18 | 0.6667 | 0.6172 |
| txt_negative_label | 1 | 102 | 0.6471 | 0.5685 |
| txt_negative_confidence_bin | high | 102 | 0.6471 | 0.5685 |
| report_length_bin | q4_high | 69 | 0.6111 | 0.5364 |
| txt_morphology_label | 1 | 246 | 0.5766 | 0.5375 |
| matched_morphology_terms_present | present | 246 | 0.5766 | 0.5375 |
| selected_n_visits_bin | high | 186 | 0.5699 | 0.5277 |
| used_images_bin | high | 186 | 0.5699 | 0.5277 |
| has_bio | 1 | 282 | 0.5319 | 0.5120 |

## Shortcut Audit Strata

These bins are audit-only and are not causal evidence.

| field | bin | n | error_rate | fn_rate | fp_rate | mean_pred_prob | mean_label | auc_if_defined |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| report_length | q4_high | 69 | 0.4928 | 0.3636 | 0.6111 | 0.5535 | 0.4783 | 0.5438 |
| bio_missing_count | 0 | 18 | 0.4444 | 0.2222 | 0.6667 | 0.6497 | 0.5000 | 0.5802 |
| selected_n_visits | high | 186 | 0.4140 | 0.2581 | 0.5699 | 0.5838 | 0.5000 | 0.6412 |
| used_images | high | 186 | 0.4140 | 0.2581 | 0.5699 | 0.5838 | 0.5000 | 0.6412 |
| report_length | q2_midlow | 69 | 0.3913 | 0.2333 | 0.5128 | 0.5724 | 0.4348 | 0.7282 |
| image_padding_count | high | 282 | 0.3723 | 0.2128 | 0.5319 | 0.5886 | 0.5000 | 0.7015 |
| has_bio | 1 | 282 | 0.3723 | 0.2128 | 0.5319 | 0.5886 | 0.5000 | 0.7015 |
| bio_missing_count | 2 | 264 | 0.3674 | 0.2121 | 0.5227 | 0.5845 | 0.5000 | 0.7088 |
| report_length | q3_midhigh | 72 | 0.3333 | 0.2051 | 0.4848 | 0.6058 | 0.5417 | 0.7094 |
| selected_n_visits | low | 96 | 0.2917 | 0.1250 | 0.4583 | 0.5980 | 0.5000 | 0.8138 |
| used_images | low | 96 | 0.2917 | 0.1250 | 0.4583 | 0.5980 | 0.5000 | 0.8138 |
| report_length | q1_low | 72 | 0.2778 | 0.0769 | 0.5152 | 0.6206 | 0.5417 | 0.7949 |

## Test Reporting-Only

Test reporting-only strata rows: 21.
