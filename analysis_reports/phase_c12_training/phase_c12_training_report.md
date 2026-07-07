# Phase C12 Single-Seed Training Pilot Report

Phase C12 trained a single-seed pilot on the deterministic report-filter manifest. This report is validation-driven; test metrics are reporting-only.

## Validation Metrics

| metric | value |
| --- | ---: |
| seed | 0 |
| best_epoch | 25 |
| AUC | 0.7936 |
| AUPRC | 0.8055 |
| ACC@0.5 | 0.6809 |
| sensitivity@0.5 | 0.5745 |
| specificity@0.5 | 0.7872 |
| FN / FP | 20 / 10 |
| positive-negative probability gap | 0.2047 |

## Error Pattern

| validation error type | n_errors | note |
| --- | ---: | --- |
| morphology_positive_false_negative | 18 | Main remaining failure mode. |
| borderline_error | 5 | Mixed FN/FP near threshold. |
| long_report_or_multivisit_uncertainty | 3 | Remaining report aggregation risk. |
| other_error | 3 | Needs case review if stress seeds pass. |
| morphology_low_confidence_false_positive | 1 | FP residual is much smaller than C8/C9. |

## Shortcut Residual Audit

| split | max_abs_spearman | linear_r2_prob_from_shortcuts | shortcut_only_label_auc_audit_only |
| --- | ---: | ---: | ---: |
| val | 0.2394 | 0.0821 | 0.3332 |
| test_reporting_only | 0.3562 | 0.1055 | 0.2883 |

## Decision

`ALLOW_C12_STRESS_SEED_PILOT`.

The single-seed validation AUC improved relative to the strict MVP C8 route, and shortcut-only label AUC remains below chance in the audit. However, the result is not sufficient for promotion because validation sensitivity is low and false negatives dominate. Run stress seeds on the same manifest before any formal selection or model claim.
