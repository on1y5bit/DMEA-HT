# Strict MVP Evidence Diagnostics Report

Target model: strict structural matched DMEA-MVP. Evidence fields and shortcut fields are analysis-only.

## Undefined Validation AUC/AUPRC Strata

_No rows._

## Validation Strata With Highest False-Negative Rates

| stratum_name | stratum_value | n | false_negative_rate | mean_pred_prob_positive |
| --- | --- | --- | --- | --- |
| txt_negative_confidence_bin | medium | 19 | 1.0000 | 0.3023 |
| txt_morphology_label | 0 | 12 | 1.0000 | 0.4727 |
| txt_morphology_confidence_bin | low | 12 | 1.0000 | 0.4727 |
| matched_morphology_terms_present | absent | 12 | 1.0000 | 0.4727 |
| report_length_bin | q4_high | 24 | 0.6667 | 0.4191 |
| bio_missing_count_bin | 0 | 6 | 0.6667 | 0.6184 |
| selected_n_visits_bin | high | 62 | 0.5806 | 0.4802 |
| used_images_bin | high | 62 | 0.5806 | 0.4802 |
| txt_negative_label | 0 | 60 | 0.5667 | 0.5224 |
| report_length_bin | q2_midlow | 23 | 0.5000 | 0.5483 |

## Validation Strata With Highest False-Positive Rates

| stratum_name | stratum_value | n | false_positive_rate | mean_pred_prob_negative |
| --- | --- | --- | --- | --- |
| bio_missing_count_bin | 0 | 6 | 0.6667 | 0.5432 |
| txt_morphology_confidence_bin | high | 38 | 0.3333 | 0.4169 |
| txt_negative_label | 1 | 34 | 0.2941 | 0.4131 |
| txt_negative_confidence_bin | high | 34 | 0.2941 | 0.4131 |
| txt_negative_confidence_bin | low | 41 | 0.2778 | 0.3646 |
| report_length_bin | q1_low | 24 | 0.2727 | 0.3698 |
| report_length_bin | q4_high | 24 | 0.2500 | 0.3712 |
| txt_morphology_label | 1 | 82 | 0.2432 | 0.3772 |
| matched_morphology_terms_present | present | 82 | 0.2432 | 0.3772 |
| selected_n_visits_bin | high | 62 | 0.2258 | 0.3631 |

## Shortcut Audit Strata

These bins are audit-only and are not causal evidence.

| field | bin | n | error_rate | fn_rate | fp_rate | mean_pred_prob | mean_label | auc_if_defined |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bio_missing_count | 0 | 6 | 0.6667 | 0.6667 | 0.6667 | 0.5808 | 0.5000 | 0.5556 |
| report_length | q4_high | 24 | 0.4583 | 0.6667 | 0.2500 | 0.3951 | 0.5000 | 0.6111 |
| selected_n_visits | high | 62 | 0.4032 | 0.5806 | 0.2258 | 0.4217 | 0.5000 | 0.7045 |
| used_images | high | 62 | 0.4032 | 0.5806 | 0.2258 | 0.4217 | 0.5000 | 0.7045 |
| report_length | q3_midhigh | 23 | 0.3478 | 0.5000 | 0.1818 | 0.4401 | 0.5217 | 0.7424 |
| image_padding_count | high | 94 | 0.3191 | 0.4255 | 0.2128 | 0.4591 | 0.5000 | 0.7936 |
| has_bio | 1 | 94 | 0.3191 | 0.4255 | 0.2128 | 0.4591 | 0.5000 | 0.7936 |
| report_length | q2_midlow | 23 | 0.3043 | 0.5000 | 0.1538 | 0.4243 | 0.4348 | 0.8923 |
| bio_missing_count | 2 | 88 | 0.2955 | 0.4091 | 0.1818 | 0.4509 | 0.5000 | 0.8073 |
| report_length | q1_low | 24 | 0.1667 | 0.0769 | 0.2727 | 0.5748 | 0.5417 | 0.9441 |
| selected_n_visits | low | 32 | 0.1562 | 0.1250 | 0.1875 | 0.5317 | 0.5000 | 0.9453 |
| used_images | low | 32 | 0.1562 | 0.1250 | 0.1875 | 0.5317 | 0.5000 | 0.9453 |

## Test Reporting-Only

Test reporting-only strata rows: 21.
