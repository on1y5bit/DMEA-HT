# Phase C13 FN Recall Audit

This is an analysis-only follow-up to C12. It uses validation errors to identify recall bottlenecks before any new training pilot.

## Validation Error Balance

- Validation errors: 30.
- Validation FN / FP: 20 / 10.

## Error Type Summary

| split | confusion_type | error_type | n | mean_pred_prob | mean_abs_error | mean_report_length | mean_selected_n_visits |
| --- | --- | --- | --- | --- | --- | --- | --- |
| val | FN | morphology_positive_false_negative | 18 | 0.3614 | 0.6386 | 826.6667 | 3.1667 |
| val | FN | borderline_error | 2 | 0.4727 | 0.5273 | 185.0000 | 1.5000 |
| val | FP | borderline_error | 3 | 0.5374 | 0.5374 | 180.3333 | 1.6667 |
| val | FP | long_report_or_multivisit_uncertainty | 3 | 0.5670 | 0.5670 | 1362.6667 | 4.0000 |
| val | FP | other_error | 3 | 0.6891 | 0.6891 | 215.0000 | 1.3333 |
| val | FP | morphology_low_confidence_false_positive | 1 | 0.7055 | 0.7055 | 829.0000 | 4.0000 |

## FN Feature Summary

| feature | value | n_fn | mean_pred_prob | mean_abs_error | mean_report_length | mean_selected_n_visits |
| --- | --- | --- | --- | --- | --- | --- |
| bio_missing_count | 2 | 18 | 0.3618 | 0.6382 | 812.2778 | 3.1111 |
| bio_missing_count | 0 | 2 | 0.4688 | 0.5312 | 314.5000 | 2.0000 |
| morphology_confidence | 1.0000 | 13 | 0.3507 | 0.6493 | 898.8462 | 3.3846 |
| morphology_confidence | 0.7000 | 5 | 0.3892 | 0.6108 | 639.0000 | 2.6000 |
| morphology_confidence | 0.0000 | 2 | 0.4727 | 0.5273 | 185.0000 | 1.5000 |
| negative_confidence | 0.0000 | 10 | 0.4048 | 0.5952 | 584.5000 | 2.7000 |
| negative_confidence | 0.5000 | 7 | 0.3023 | 0.6977 | 916.0000 | 3.2857 |
| negative_confidence | 1.0000 | 3 | 0.4286 | 0.5714 | 997.6667 | 3.3333 |
| negative_label | 0 | 17 | 0.3626 | 0.6374 | 721.0000 | 2.9412 |
| negative_label | 1 | 3 | 0.4286 | 0.5714 | 997.6667 | 3.3333 |
| report_length_bin | q4_high | 8 | 0.3176 | 0.6824 | 1304.7500 | 4.3750 |
| report_length_bin | q3_midhigh | 6 | 0.3618 | 0.6382 | 523.0000 | 2.3333 |
| report_length_bin | q2_midlow | 5 | 0.4512 | 0.5488 | 311.8000 | 2.0000 |
| report_length_bin | q1_low | 1 | 0.4820 | 0.5180 | 115.0000 | 1.0000 |
| selected_n_visits_exact | 2 | 7 | 0.3819 | 0.6181 | 379.5714 | 2.0000 |
| selected_n_visits_exact | 3 | 6 | 0.3876 | 0.6124 | 704.5000 | 3.0000 |
| selected_n_visits_exact | 1 | 2 | 0.4401 | 0.5599 | 247.5000 | 1.0000 |
| selected_n_visits_exact | 5 | 2 | 0.2443 | 0.7557 | 1366.5000 | 5.0000 |
| selected_n_visits_exact | 6 | 2 | 0.3450 | 0.6550 | 1923.0000 | 6.0000 |
| selected_n_visits_exact | 4 | 1 | 0.3928 | 0.6072 | 1292.0000 | 4.0000 |

## Evidence Strata Relevant To Recall

### Morphology Confidence

| stratum_name | stratum_value | n | n_positive | n_negative | auc_if_defined | sensitivity_at_0p5 | specificity_at_0p5 | false_negative_count | false_negative_rate | false_positive_count | false_positive_rate | positive_negative_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| txt_morphology_confidence_bin | high | 38 | 29 | 9 | 0.6973 | 0.5517 | 0.6667 | 13 | 0.4483 | 3 | 0.3333 | 0.1336 |
| txt_morphology_confidence_bin | medium | 44 | 16 | 28 | 0.8482 | 0.6875 | 0.7857 | 5 | 0.3125 | 6 | 0.2143 | 0.2282 |
| txt_morphology_confidence_bin | low | 12 | 2 | 10 | 0.9000 | 0.0000 | 0.9000 | 2 | 1.0000 | 1 | 0.1000 | 0.1912 |

### Negative Evidence Label

| stratum_name | stratum_value | n | n_positive | n_negative | auc_if_defined | sensitivity_at_0p5 | specificity_at_0p5 | false_negative_count | false_negative_rate | false_positive_count | false_positive_rate | positive_negative_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| txt_negative_label | 0 | 60 | 30 | 30 | 0.7811 | 0.4333 | 0.8333 | 17 | 0.5667 | 5 | 0.1667 | 0.1975 |
| txt_negative_label | 1 | 34 | 17 | 17 | 0.8443 | 0.8235 | 0.7059 | 3 | 0.1765 | 5 | 0.2941 | 0.2174 |

### Negative Evidence Confidence

| stratum_name | stratum_value | n | n_positive | n_negative | auc_if_defined | sensitivity_at_0p5 | specificity_at_0p5 | false_negative_count | false_negative_rate | false_positive_count | false_positive_rate | positive_negative_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| txt_negative_confidence_bin | low | 41 | 23 | 18 | 0.8019 | 0.5652 | 0.7222 | 10 | 0.4348 | 5 | 0.2778 | 0.2248 |
| txt_negative_confidence_bin | medium | 19 | 7 | 12 | 0.6667 | 0.0000 | 1.0000 | 7 | 1.0000 | 0 | 0.0000 | 0.0369 |
| txt_negative_confidence_bin | high | 34 | 17 | 17 | 0.8443 | 0.8235 | 0.7059 | 3 | 0.1765 | 5 | 0.2941 | 0.2174 |

### Report Length

| stratum_name | stratum_value | n | n_positive | n_negative | auc_if_defined | sensitivity_at_0p5 | specificity_at_0p5 | false_negative_count | false_negative_rate | false_positive_count | false_positive_rate | positive_negative_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| report_length_bin | q4_high | 24 | 12 | 12 | 0.6111 | 0.3333 | 0.7500 | 8 | 0.6667 | 3 | 0.2500 | 0.0479 |
| report_length_bin | q3_midhigh | 23 | 12 | 11 | 0.7424 | 0.5000 | 0.8182 | 6 | 0.5000 | 2 | 0.1818 | 0.1516 |
| report_length_bin | q2_midlow | 23 | 10 | 13 | 0.8923 | 0.5000 | 0.8462 | 5 | 0.5000 | 2 | 0.1538 | 0.2193 |
| report_length_bin | q1_low | 24 | 13 | 11 | 0.9441 | 0.9231 | 0.7273 | 1 | 0.0769 | 3 | 0.2727 | 0.3785 |

### Selected Visits

| stratum_name | stratum_value | n | n_positive | n_negative | auc_if_defined | sensitivity_at_0p5 | specificity_at_0p5 | false_negative_count | false_negative_rate | false_positive_count | false_positive_rate | positive_negative_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| selected_n_visits_bin | high | 62 | 31 | 31 | 0.7045 | 0.4194 | 0.7742 | 18 | 0.5806 | 7 | 0.2258 | 0.1171 |
| selected_n_visits_bin | low | 32 | 16 | 16 | 0.9453 | 0.8750 | 0.8125 | 2 | 0.1250 | 3 | 0.1875 | 0.3744 |

## C12 Filter Positive-Damage Check

| split | label | n | n_filtered | n_morphology_changed | n_image_weak_changed |
| --- | --- | --- | --- | --- | --- |
| test | 0 | 42 | 5 | 3 | 3 |
| test | 1 | 42 | 3 | 1 | 1 |
| train | 0 | 301 | 16 | 8 | 8 |
| train | 1 | 301 | 16 | 5 | 5 |
| val | 0 | 47 | 4 | 2 | 2 |
| val | 1 | 47 | 0 | 0 | 0 |

## Lowest-Probability Validation FN Cases

| patient_id | pred_prob | confidence_group | error_type | txt_morphology_confidence | txt_negative_label | txt_negative_confidence | selected_n_visits | report_length | report_length_bin | n_dropped_clauses | latest_diffuse_ht_like | changed_txt_morphology_label | matched_morphology_terms | matched_negative_terms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10042173 | 0.1648 | high_confidence_negative | morphology_positive_false_negative | 1.0000 | 0 | 0.5000 | 5 | 1458 | q4_high | 0 | 1 | 0 | 实质回声不均/回声不均/回声欠均/回声欠均匀/低回声 | ['未见异常血流'] |
| 10132146 | 0.2035 | medium_confidence_negative | morphology_positive_false_negative | 1.0000 | 0 | 0.0000 | 2 | 422 | q3_midhigh | 0 | 1 | 0 | 实质回声不均/回声不均/回声欠均/回声欠均匀/低回声 | [] |
| 10014141 | 0.2216 | medium_confidence_negative | morphology_positive_false_negative | 1.0000 | 0 | 0.5000 | 3 | 894 | q4_high | 0 | 1 | 0 | 实质回声不均/回声不均/回声欠均/回声欠均匀/低回声 | ['未见异常血流'] |
| 10127720 | 0.2826 | medium_confidence_negative | morphology_positive_false_negative | 1.0000 | 0 | 0.5000 | 2 | 483 | q3_midhigh | 0 | 1 | 0 | 实质回声不均/回声不均/回声欠均/回声欠均匀/低回声 | ['未见异常血流'] |
| 10003245 | 0.2865 | medium_confidence_negative | morphology_positive_false_negative | 0.7000 | 0 | 0.0000 | 3 | 911 | q4_high | 0 | 1 | 0 | 回声不均 | [] |
| 10132330 | 0.3237 | medium_confidence_negative | morphology_positive_false_negative | 0.7000 | 0 | 0.0000 | 5 | 1275 | q4_high | 0 | 0 | 0 | 回声欠均/回声欠均匀/低回声 | [] |
| 10151451 | 0.3252 | medium_confidence_negative | morphology_positive_false_negative | 1.0000 | 0 | 0.5000 | 2 | 506 | q3_midhigh | 0 | 1 | 0 | 实质回声不均/回声不均/回声欠均/回声欠均匀/低回声 | ['未见异常血流'] |
| 10082765 | 0.3311 | medium_confidence_negative | morphology_positive_false_negative | 1.0000 | 0 | 0.5000 | 6 | 1399 | q4_high | 0 | 1 | 0 | 实质回声不均/回声不均/回声欠均/回声欠均匀/低回声 | ['未见异常血流'] |
| 10064537 | 0.3590 | medium_confidence_negative | morphology_positive_false_negative | 1.0000 | 1 | 1.0000 | 6 | 2447 | q4_high | 0 | 1 | 0 | 实质回声不均/回声不均/回声欠均/回声欠均匀/低回声 | ['实质回声均匀', '回声均匀', '未见异常血流'] |
| 10064351 | 0.3928 | medium_confidence_negative | morphology_positive_false_negative | 1.0000 | 0 | 0.5000 | 4 | 1292 | q4_high | 0 | 1 | 0 | 实质回声不均/回声不均/回声欠均/回声欠均匀/低回声 | ['未见异常血流'] |
| 10110001 | 0.3981 | medium_confidence_negative | morphology_positive_false_negative | 0.7000 | 0 | 0.5000 | 1 | 380 | q2_midlow | 0 | 1 | 0 | 回声欠均/回声欠均匀/低回声 | ['未见异常血流'] |
| 10157441 | 0.4290 | borderline | morphology_positive_false_negative | 1.0000 | 0 | 0.0000 | 3 | 638 | q3_midhigh | 0 | 1 | 0 | 实质回声不均/回声不均/回声欠均/回声欠均匀/低回声 | [] |
| 10093113 | 0.4589 | borderline | morphology_positive_false_negative | 1.0000 | 0 | 0.0000 | 3 | 644 | q3_midhigh | 0 | 1 | 0 | 实质回声不均/回声不均/低回声 | [] |
| 10066912 | 0.4607 | borderline | morphology_positive_false_negative | 1.0000 | 1 | 1.0000 | 2 | 362 | q2_midlow | 0 | 1 | 0 | 实质回声不均/回声不均/回声欠均/回声欠均匀/低回声 | ['回声均匀'] |
| 10131359 | 0.4613 | borderline | morphology_positive_false_negative | 1.0000 | 0 | 0.0000 | 3 | 762 | q4_high | 0 | 1 | 0 | 实质回声不均/回声不均/回声欠均/回声欠均匀/低回声 | [] |
| 10151887 | 0.4633 | borderline | borderline_error | 0.0000 | 0 | 0.0000 | 2 | 255 | q2_midlow | 0 | 1 | 0 | [] | [] |
| 10098069 | 0.4660 | borderline | morphology_positive_false_negative | 0.7000 | 1 | 1.0000 | 2 | 184 | q2_midlow | 0 | 0 | 0 | 回声欠均/回声欠均匀/低回声 | ['回声均匀'] |
| 10064626 | 0.4679 | borderline | morphology_positive_false_negative | 1.0000 | 0 | 0.0000 | 3 | 378 | q2_midlow | 0 | 1 | 0 | 实质回声不均/回声不均/回声欠均/回声欠均匀/低回声 | [] |
| 10135408 | 0.4716 | borderline | morphology_positive_false_negative | 0.7000 | 0 | 0.0000 | 2 | 445 | q3_midhigh | 0 | 1 | 0 | 回声不均/回声欠均/回声欠均匀/低回声 | [] |
| 10012205 | 0.4820 | borderline | borderline_error | 0.0000 | 0 | 0.0000 | 1 | 115 | q1_low | 0 | 0 | 0 | [] | [] |

## Interpretation

- C12 reduced false positives, but validation false negatives now dominate.
- C12 manifest audit showed no validation-positive report filtering or morphology-label damage, so the FN pattern is not explained by C12 deleting positive validation evidence.
- FN concentration in long reports or high-visit patients points to temporal/report aggregation as the next recall bottleneck.
- Negative evidence is not sufficient as a global explanation because many FNs have no strong negative label.

## Recommendation

`DESIGN_C13_TEMPORAL_OR_LONG_REPORT_RECALL_PILOT_AFTER_STRESS_SEEDS`.

Stress-seed results should be collected before launching any C13 training pilot.
