# Strict MVP Evidence Diagnostics Report

Target model: strict structural matched DMEA-MVP. Evidence fields and shortcut fields are analysis-only.

## Undefined Validation AUC/AUPRC Strata

_No rows._

## Validation Strata With Highest False-Negative Rates

| stratum_name | stratum_value | n | false_negative_rate | mean_pred_prob_positive |
| --- | --- | --- | --- | --- |
| txt_morphology_label | 0 | 36 | 0.8333 | 0.3788 |
| txt_morphology_confidence_bin | low | 36 | 0.8333 | 0.3788 |
| matched_morphology_terms_present | absent | 36 | 0.8333 | 0.3788 |
| txt_negative_confidence_bin | medium | 57 | 0.6667 | 0.4000 |
| report_length_bin | q4_high | 69 | 0.4848 | 0.4914 |
| txt_morphology_confidence_bin | medium | 132 | 0.4167 | 0.5470 |
| txt_negative_label | 0 | 180 | 0.3889 | 0.5832 |
| selected_n_visits_bin | high | 186 | 0.3871 | 0.5690 |
| used_images_bin | high | 186 | 0.3871 | 0.5690 |
| bio_missing_count_bin | 2plus | 264 | 0.3712 | 0.5892 |

## Validation Strata With Highest False-Positive Rates

| stratum_name | stratum_value | n | false_positive_rate | mean_pred_prob_negative |
| --- | --- | --- | --- | --- |
| txt_morphology_confidence_bin | high | 114 | 0.5556 | 0.5122 |
| bio_missing_count_bin | 0 | 18 | 0.5556 | 0.4223 |
| report_length_bin | q4_high | 69 | 0.1944 | 0.3150 |
| txt_negative_label | 1 | 102 | 0.1765 | 0.3021 |
| txt_negative_confidence_bin | high | 102 | 0.1765 | 0.3021 |
| txt_morphology_label | 1 | 246 | 0.1712 | 0.2785 |
| matched_morphology_terms_present | present | 246 | 0.1712 | 0.2785 |
| txt_negative_confidence_bin | low | 123 | 0.1667 | 0.2265 |
| report_length_bin | q1_low | 75 | 0.1667 | 0.2539 |
| report_length_bin | q2_midlow | 66 | 0.1667 | 0.2449 |

## Shortcut Audit Strata

These bins are audit-only and are not causal evidence.

| field | bin | n | error_rate | fn_rate | fp_rate | mean_pred_prob | mean_label | auc_if_defined |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| report_length | q4_high | 69 | 0.3333 | 0.4848 | 0.1944 | 0.3994 | 0.4783 | 0.7449 |
| bio_missing_count | 0 | 18 | 0.2778 | 0.0000 | 0.5556 | 0.6034 | 0.5000 | 0.8765 |
| selected_n_visits | high | 186 | 0.2742 | 0.3871 | 0.1613 | 0.4223 | 0.5000 | 0.8456 |
| used_images | high | 186 | 0.2742 | 0.3871 | 0.1613 | 0.4223 | 0.5000 | 0.8456 |
| report_length | q2_midlow | 66 | 0.2576 | 0.3667 | 0.1667 | 0.3826 | 0.4545 | 0.7917 |
| image_padding_count | high | 282 | 0.2482 | 0.3475 | 0.1489 | 0.4294 | 0.5000 | 0.8581 |
| has_bio | 1 | 282 | 0.2482 | 0.3475 | 0.1489 | 0.4294 | 0.5000 | 0.8581 |
| bio_missing_count | 2 | 264 | 0.2462 | 0.3712 | 0.1212 | 0.4175 | 0.5000 | 0.8633 |
| report_length | q3_midhigh | 72 | 0.2222 | 0.3590 | 0.0606 | 0.4309 | 0.5417 | 0.9425 |
| selected_n_visits | low | 96 | 0.1979 | 0.2708 | 0.1250 | 0.4431 | 0.5000 | 0.8733 |
| used_images | low | 96 | 0.1979 | 0.2708 | 0.1250 | 0.4431 | 0.5000 | 0.8733 |
| report_length | q1_low | 75 | 0.1867 | 0.2051 | 0.1667 | 0.4966 | 0.5200 | 0.9074 |

## Test Reporting-Only

Test reporting-only strata rows: 21.
