# Strict MVP Error Taxonomy Report

Target model: strict structural matched DMEA-MVP. This is analysis-only; no training was performed.

## Validation Error Taxonomy

| split | error_type | n_errors | proportion_of_errors | false_negative_count | false_positive_count | mean_pred_prob | mean_abs_error |
| --- | --- | --- | --- | --- | --- | --- | --- |
| val | morphology_positive_false_negative | 44 | 0.6286 | 44 | 0 | 0.3504 | 0.6496 |
| val | long_report_or_multivisit_uncertainty | 7 | 0.1000 | 0 | 7 | 0.5993 | 0.5993 |
| val | other_error | 7 | 0.1000 | 3 | 4 | 0.5228 | 0.7347 |
| val | borderline_error | 6 | 0.0857 | 2 | 4 | 0.4947 | 0.5418 |
| val | high_confidence_false_positive | 4 | 0.0571 | 0 | 4 | 0.8563 | 0.8563 |
| val | morphology_low_confidence_false_positive | 2 | 0.0286 | 0 | 2 | 0.6124 | 0.6124 |

## Top False-Negative Categories

| error_type | n |
| --- | --- |
| morphology_positive_false_negative | 44 |
| other_error | 3 |
| borderline_error | 2 |

## Top False-Positive Categories

| error_type | n |
| --- | --- |
| long_report_or_multivisit_uncertainty | 7 |
| high_confidence_false_positive | 4 |
| other_error | 4 |
| borderline_error | 4 |
| morphology_low_confidence_false_positive | 2 |

## Test Reporting-Only Note

Test reporting-only error rows: 61. These rows are for transparency and manual review only.
