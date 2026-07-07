# Strict MVP Error Taxonomy Report

Target model: strict structural matched DMEA-MVP. This is analysis-only; no training was performed.

## Validation Error Taxonomy

| split | error_type | n_errors | proportion_of_errors | false_negative_count | false_positive_count | mean_pred_prob | mean_abs_error |
| --- | --- | --- | --- | --- | --- | --- | --- |
| val | morphology_positive_false_negative | 18 | 0.6000 | 18 | 0 | 0.3614 | 0.6386 |
| val | borderline_error | 5 | 0.1667 | 2 | 3 | 0.5115 | 0.5334 |
| val | long_report_or_multivisit_uncertainty | 3 | 0.1000 | 0 | 3 | 0.5670 | 0.5670 |
| val | other_error | 3 | 0.1000 | 0 | 3 | 0.6891 | 0.6891 |
| val | morphology_low_confidence_false_positive | 1 | 0.0333 | 0 | 1 | 0.7055 | 0.7055 |

## Top False-Negative Categories

| error_type | n |
| --- | --- |
| morphology_positive_false_negative | 18 |
| borderline_error | 2 |

## Top False-Positive Categories

| error_type | n |
| --- | --- |
| other_error | 3 |
| long_report_or_multivisit_uncertainty | 3 |
| borderline_error | 3 |
| morphology_low_confidence_false_positive | 1 |

## Test Reporting-Only Note

Test reporting-only error rows: 26. These rows are for transparency and manual review only.
