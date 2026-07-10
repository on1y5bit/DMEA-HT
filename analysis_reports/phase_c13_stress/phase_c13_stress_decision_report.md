# Phase C13 Temporal-Focus Stress-Seed Decision Report

This report is validation-selected. Test metrics are reporting-only and were not used for model selection.

## Inputs

- Run directory: `runs/dmea_ht_v2_c13_temporal_focus_stress_seeds`
- Manifest: `/data/csb/DMEA-HT/HT_2025.12_25/manifest_distmatch_structmatch_evidence_v2_c13_temporal_focus.jsonl`
- Seeds: `[0, 42, 3407]`
- Primary metric: validation AUC

## Validation Metrics

| seed | best_epoch | AUC | AUPRC | ACC | Sensitivity | Specificity | FP | FN |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 16 | 0.8656 | 0.8615 | 0.7128 | 0.5319 | 0.8936 | 5 | 22 |
| 42 | 17 | 0.8746 | 0.8579 | 0.8191 | 0.8298 | 0.8085 | 9 | 8 |
| 3407 | 15 | 0.8592 | 0.8518 | 0.7234 | 0.5957 | 0.8511 | 7 | 19 |

Summary:

- Validation AUC mean / std: 0.8665 / 0.0077.
- Validation AUPRC mean / std: 0.8570 / 0.0049.
- Validation sensitivity mean / std: 0.6525 / 0.1568.
- Validation specificity mean / std: 0.8511 / 0.0426.

## Test Reporting-Only Metrics

| seed | best_epoch | AUC | AUPRC | ACC | Sensitivity | Specificity | FP | FN |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 16 | 0.8549 | 0.8188 | 0.7619 | 0.6429 | 0.8810 | 5 | 15 |
| 42 | 17 | 0.8418 | 0.8090 | 0.7619 | 0.8095 | 0.7143 | 12 | 8 |
| 3407 | 15 | 0.8413 | 0.8235 | 0.7500 | 0.7143 | 0.7857 | 9 | 12 |

## Validation Error Taxonomy

| error_type | n_errors | FN | FP | proportion |
| --- | ---: | ---: | ---: | ---: |
| morphology_positive_false_negative | 44 | 44 | 0 | 0.6286 |
| long_report_or_multivisit_uncertainty | 7 | 0 | 7 | 0.1000 |
| other_error | 7 | 3 | 4 | 0.1000 |
| borderline_error | 6 | 2 | 4 | 0.0857 |
| high_confidence_false_positive | 4 | 0 | 4 | 0.0571 |
| morphology_low_confidence_false_positive | 2 | 0 | 2 | 0.0286 |

## Shortcut Residual Audit

| split | max abs Spearman | linear R2 | shortcut-only label AUC audit-only |
| --- | ---: | ---: | ---: |
| val | 0.1549 | 0.0601 | 0.4762 |
| test_reporting_only | 0.1745 | 0.0587 | 0.3983 |

## Decision

- Recommendation: `PROMOTE_C13_AS_CURRENT_STRICT_BEST_NOT_FINAL`.
- C13 is more stable than C12 under the requested seeds `[0, 42, 3407]`.
- The structural shortcut residual audit remains acceptable and does not suggest selected visit/image/bio/report-length fields alone recover labels.
- C13 still misses the target validation AUC of 0.90.
- The next iteration should target morphology-positive false negatives and long-report high-length recall without feeding shortcut fields into the classifier or changing labels/splits/task definition.
