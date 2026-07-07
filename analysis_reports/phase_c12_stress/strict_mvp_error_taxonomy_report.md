# Strict MVP Error Taxonomy Report

Target model: strict structural matched DMEA-MVP. This is analysis-only; no training was performed.

## Validation Error Taxonomy

| split | error_type | n_errors | proportion_of_errors | false_negative_count | false_positive_count | mean_pred_prob | mean_abs_error |
| --- | --- | --- | --- | --- | --- | --- | --- |
| val | morphology_positive_false_negative | 28 | 0.2667 | 28 | 0 | 0.3838 | 0.6162 |
| val | long_report_or_multivisit_uncertainty | 22 | 0.2095 | 0 | 22 | 0.6354 | 0.6354 |
| val | other_error | 22 | 0.2095 | 1 | 21 | 0.6914 | 0.7017 |
| val | high_confidence_false_positive | 13 | 0.1238 | 0 | 13 | 0.8460 | 0.8460 |
| val | morphology_low_confidence_false_positive | 11 | 0.1048 | 0 | 11 | 0.6685 | 0.6685 |
| val | borderline_error | 9 | 0.0857 | 1 | 8 | 0.5493 | 0.5619 |

## Top False-Negative Categories

| error_type | n |
| --- | --- |
| morphology_positive_false_negative | 28 |
| other_error | 1 |
| borderline_error | 1 |

## Top False-Positive Categories

| error_type | n |
| --- | --- |
| long_report_or_multivisit_uncertainty | 22 |
| other_error | 21 |
| high_confidence_false_positive | 13 |
| morphology_low_confidence_false_positive | 11 |
| borderline_error | 8 |

## Test Reporting-Only Note

Test reporting-only error rows: 87. These rows are for transparency and manual review only.
