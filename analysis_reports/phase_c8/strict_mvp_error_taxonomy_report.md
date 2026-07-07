# Strict MVP Error Taxonomy Report

Target model: strict structural matched DMEA-MVP. This is analysis-only; no training was performed.

## Validation Error Taxonomy

| split | error_type | n_errors | proportion_of_errors | false_negative_count | false_positive_count | mean_pred_prob | mean_abs_error |
| --- | --- | --- | --- | --- | --- | --- | --- |
| val | long_report_or_multivisit_uncertainty | 25 | 0.2604 | 0 | 25 | 0.6754 | 0.6754 |
| val | other_error | 25 | 0.2604 | 1 | 24 | 0.6786 | 0.6882 |
| val | morphology_positive_false_negative | 15 | 0.1562 | 15 | 0 | 0.3884 | 0.6116 |
| val | high_confidence_false_positive | 12 | 0.1250 | 0 | 12 | 0.8639 | 0.8639 |
| val | morphology_low_confidence_false_positive | 10 | 0.1042 | 0 | 10 | 0.6412 | 0.6412 |
| val | borderline_error | 9 | 0.0938 | 0 | 9 | 0.5466 | 0.5466 |

## Top False-Negative Categories

| error_type | n |
| --- | --- |
| morphology_positive_false_negative | 15 |
| other_error | 1 |

## Top False-Positive Categories

| error_type | n |
| --- | --- |
| long_report_or_multivisit_uncertainty | 25 |
| other_error | 24 |
| high_confidence_false_positive | 12 |
| morphology_low_confidence_false_positive | 10 |
| borderline_error | 9 |

## Test Reporting-Only Note

Test reporting-only error rows: 84. These rows are for transparency and manual review only.
