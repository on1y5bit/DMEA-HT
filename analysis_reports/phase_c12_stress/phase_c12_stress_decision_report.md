# Phase C12 Stress-Seed Decision Report

Phase C12 stress seeds evaluate whether the report-filter manifest improvement is stable before any formal model claim.

## Validation Metrics

| seed | AUC | AUPRC | sensitivity | specificity | FN | FP | best_epoch |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.7773 | 0.8057 | 0.5957 | 0.7447 | 19 | 12 | 14 |
| 3 | 0.7691 | 0.7557 | 0.8085 | 0.5532 | 9 | 21 | 14 |
| 42 | 0.7429 | 0.7767 | 0.9574 | 0.1064 | 2 | 42 | 7 |

## Stress Summary

| split | AUC mean | AUC std | AUPRC mean | AUPRC std | sensitivity mean | specificity mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| val | 0.7631 | 0.0180 | 0.7794 | 0.0251 | 0.7872 | 0.4681 |
| test_reporting_only | 0.7509 | 0.0249 | 0.6963 | 0.0362 | 0.8571 | 0.4524 |

## Error Pattern

| validation error type | n_errors | FN | FP |
| --- | ---: | ---: | ---: |
| morphology_positive_false_negative | 28 | 28 | 0 |
| long_report_or_multivisit_uncertainty | 22 | 0 | 22 |
| other_error | 22 | 1 | 21 |
| high_confidence_false_positive | 13 | 0 | 13 |
| morphology_low_confidence_false_positive | 11 | 0 | 11 |
| borderline_error | 9 | 1 | 8 |

## Shortcut Residual Audit

| split | pooled max_abs_spearman | pooled linear_r2_prob_from_shortcuts | pooled shortcut_only_label_auc_audit_only |
| --- | ---: | ---: | ---: |
| val | 0.0946 | 0.0373 | 0.4918 |
| test_reporting_only | 0.1036 | 0.0286 | 0.3746 |

## Decision

`DO_NOT_PROMOTE_C12_FORMALLY`.

C12 is a useful report-construction direction because it modestly improves validation AUC and shortcut residuals are low, but it is not stable enough for formal selection. The stress seeds expose a sensitivity-specificity swing, especially seed 42. The next step should address long-report and multi-visit text truncation/ordering while preserving the C12 false-positive filter.

## Next Action

`DESIGN_C13_TEMPORAL_FOCUS_REPORT_PILOT`.

The C13 pilot should be data-construction only: build a manifest that keeps labels, splits, images, and bio values unchanged, preserves the C12 report filter, and places thyroid-relevant latest and diffuse/morphology clauses before the full report text so the 256-character text encoder sees the most relevant evidence first.
