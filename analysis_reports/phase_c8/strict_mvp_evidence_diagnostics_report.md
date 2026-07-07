# Strict MVP Evidence Diagnostics Report

Target model: strict structural matched DMEA-MVP. Evidence fields and shortcut fields are analysis-only.

## Undefined Validation AUC/AUPRC Strata

_No rows._

## Validation Strata With Highest False-Negative Rates

| stratum_name | stratum_value | n | false_negative_rate | mean_pred_prob_positive |
| --- | --- | --- | --- | --- |
| txt_negative_confidence_bin | medium | 57 | 0.3810 | 0.5208 |
| report_length_bin | q4_high | 69 | 0.2121 | 0.6479 |
| txt_negative_label | 0 | 180 | 0.1778 | 0.6706 |
| txt_morphology_label | 0 | 30 | 0.1667 | 0.5819 |
| txt_morphology_confidence_bin | low | 30 | 0.1667 | 0.5819 |
| matched_morphology_terms_present | absent | 30 | 0.1667 | 0.5819 |
| bio_missing_count_bin | 2plus | 264 | 0.1212 | 0.7048 |
| selected_n_visits_bin | high | 186 | 0.1183 | 0.6982 |
| used_images_bin | high | 186 | 0.1183 | 0.6982 |
| txt_negative_confidence_bin | low | 123 | 0.1159 | 0.7162 |

## Validation Strata With Highest False-Positive Rates

| stratum_name | stratum_value | n | false_positive_rate | mean_pred_prob_negative |
| --- | --- | --- | --- | --- |
| bio_missing_count_bin | 0 | 18 | 0.7778 | 0.7027 |
| txt_negative_label | 1 | 102 | 0.7255 | 0.6111 |
| txt_negative_confidence_bin | high | 102 | 0.7255 | 0.6111 |
| selected_n_visits_bin | high | 186 | 0.6559 | 0.5619 |
| used_images_bin | high | 186 | 0.6559 | 0.5619 |
| report_length_bin | q4_high | 69 | 0.6389 | 0.5561 |
| report_length_bin | q3_midhigh | 72 | 0.6061 | 0.5517 |
| txt_morphology_label | 1 | 252 | 0.5983 | 0.5489 |
| txt_morphology_confidence_bin | high | 252 | 0.5983 | 0.5489 |
| matched_morphology_terms_present | present | 252 | 0.5983 | 0.5489 |

## Shortcut Audit Strata

These bins are audit-only and are not causal evidence.

| field | bin | n | error_rate | fn_rate | fp_rate | mean_pred_prob | mean_label | auc_if_defined |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| report_length | q4_high | 69 | 0.4348 | 0.2121 | 0.6389 | 0.6000 | 0.4783 | 0.6094 |
| bio_missing_count | 0 | 18 | 0.3889 | 0.0000 | 0.7778 | 0.7385 | 0.5000 | 0.5679 |
| selected_n_visits | high | 186 | 0.3871 | 0.1183 | 0.6559 | 0.6300 | 0.5000 | 0.6809 |
| used_images | high | 186 | 0.3871 | 0.1183 | 0.6559 | 0.6300 | 0.5000 | 0.6809 |
| report_length | q2_midlow | 69 | 0.3478 | 0.1000 | 0.5385 | 0.5942 | 0.4348 | 0.7496 |
| image_padding_count | high | 282 | 0.3404 | 0.1135 | 0.5674 | 0.6194 | 0.5000 | 0.7443 |
| has_bio | 1 | 282 | 0.3404 | 0.1135 | 0.5674 | 0.6194 | 0.5000 | 0.7443 |
| bio_missing_count | 2 | 264 | 0.3371 | 0.1212 | 0.5530 | 0.6113 | 0.5000 | 0.7569 |
| report_length | q3_midhigh | 72 | 0.3333 | 0.1026 | 0.6061 | 0.6513 | 0.5417 | 0.7374 |
| selected_n_visits | low | 96 | 0.2500 | 0.1042 | 0.3958 | 0.5988 | 0.5000 | 0.8581 |
| used_images | low | 96 | 0.2500 | 0.1042 | 0.3958 | 0.5988 | 0.5000 | 0.8581 |
| report_length | q1_low | 72 | 0.2500 | 0.0513 | 0.4848 | 0.6303 | 0.5417 | 0.8563 |

## Test Reporting-Only

Test reporting-only strata rows: 20.
