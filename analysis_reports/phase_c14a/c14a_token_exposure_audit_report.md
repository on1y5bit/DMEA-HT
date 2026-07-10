# Phase C14-A FN Token Exposure Audit

This is an analysis-only audit. No training, threshold tuning, label editing, split editing, or model-code changes were performed.

## Audit Window

- The project tokenizer is character-level with special tokens.
- `text_max_length=256` means approximately 254 report characters are visible as text tokens.
- Fields named `first256_*` use this model character-token window, not a word tokenizer.

## Validation Positive Cohorts

| row_type | n_rows | n_patients | mean_pred_prob | mean_first256_morphology_term_count | mean_first256_diffuse_ht_term_count | mean_full_report_morphology_term_count | mean_full_report_diffuse_ht_term_count | mean_latest_visit_morphology_term_count | mean_latest_visit_diffuse_ht_term_count | mean_full_report_negative_term_count | mean_positive_negative_overlap | mean_report_length_chars | mean_selected_n_visits | mean_cross_seed_pred_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FN | 49 | 23 | 0.3477 | 2.6735 | 1.4490 | 3.5102 | 1.4490 | 2.3878 | 1.4286 | 3.1224 | 0.7755 | 858.7143 | 2.5306 | 0.1012 |
| TP | 92 | 40 | 0.7369 | 2.9130 | 1.6087 | 3.5109 | 1.6413 | 2.6413 | 1.5870 | 3.1630 | 0.8587 | 667.4130 | 2.2391 | 0.1082 |
| stable_fn | 19 | 19 | 0.3257 | 2.7368 | 1.4737 | 3.6316 | 1.4737 | 2.4737 | 1.4737 | 3.1053 | 0.7895 | 887.6316 | 2.5789 | 0.1013 |
| stable_tp | 28 | 28 | 0.6485 | 2.8929 | 1.6071 | 3.4286 | 1.6429 | 2.6071 | 1.5714 | 3.1786 | 0.8571 | 629.5714 | 2.1786 | 0.1087 |

## Evidence Exposure Strata

| stratum | n_rows | n_patients | fn_count | fn_rate | mean_pred_prob | mean_cross_seed_pred_std |
| --- | --- | --- | --- | --- | --- | --- |
| diffuse_exposed_first_window | 126 | 42 | 44 | 0.3492 | 0.6003 | 0.1066 |
| only_generic_morphology_exposed | 12 | 4 | 3 | 0.2500 | 0.6497 | 0.0790 |
| no_positive_thyroid_evidence_exposed | 3 | 1 | 2 | 0.6667 | 0.4687 | 0.1796 |
| positive_negative_overlap_full_report | 117 | 39 | 38 | 0.3248 | 0.6167 | 0.1002 |

## Seed FN Overlap

| comparison | fn_count | unique_fn_count | overlap_count |
| --- | --- | --- | --- |
| seed_0 | 22 | 4 | 18 |
| seed_42 | 8 | 0 | 8 |
| seed_3407 | 19 | 0 | 19 |
| seed_0_vs_seed_42 | 22 | 16 | 7 |
| seed_0_vs_seed_3407 | 22 | 5 | 18 |
| seed_42_vs_seed_3407 | 8 | 11 | 8 |
| all_seed_intersection | 7 | 0 | 7 |

## Stable FN Examples

| patient_id | cross_seed_fn_count | cross_seed_tp_count | cross_seed_pred_mean | cross_seed_pred_std | first256_morphology_term_count | first256_diffuse_ht_term_count | full_report_diffuse_ht_term_count | latest_visit_diffuse_ht_term_count | full_report_negative_term_count | positive_negative_overlap | report_length_chars | selected_n_visits | matched_morphology_terms | matched_diffuse_terms | matched_negative_terms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10117252 | 3 | 0 | 0.3387 | 0.0753 | 3 | 0 | 0 | 0 | 5 | 1 | 589 | 2 | 回声欠均/回声欠均匀/低回声 |  | 回声均匀/实质回声均匀/未见明显异常回声/大小正常/形态大小正常 |
| 10012205 | 2 | 1 | 0.4687 | 0.1796 | 0 | 0 | 0 | 0 | 3 | 0 | 268 | 1 |  |  | 未见明显异常回声/大小正常/形态大小正常 |
| 10110001 | 3 | 0 | 0.0482 | 0.0183 | 4 | 1 | 1 | 1 | 3 | 1 | 619 | 1 | 回声欠均/回声欠均匀/低回声/表面欠光滑 | 表面欠光滑 | 大小正常/形态大小正常/未见异常血流 |
| 10003245 | 3 | 0 | 0.1301 | 0.0093 | 1 | 1 | 1 | 1 | 0 | 0 | 1150 | 3 | 回声不均 | 回声不均 |  |
| 10075536 | 3 | 0 | 0.2703 | 0.0553 | 2 | 1 | 1 | 1 | 4 | 1 | 1002 | 4 | 回声不均/回声欠均/低回声 | 回声不均 | 回声均匀/实质回声均匀/大小正常/形态大小正常 |
| 10151887 | 3 | 0 | 0.2889 | 0.1023 | 1 | 1 | 1 | 1 | 0 | 0 | 494 | 2 | 弥漫性 | 弥漫性 |  |
| 10024132 | 2 | 1 | 0.3573 | 0.1655 | 2 | 1 | 1 | 1 | 2 | 1 | 362 | 1 | 回声不均/低回声 | 回声不均 | 大小正常/形态大小正常 |
| 10168610 | 2 | 1 | 0.4558 | 0.0969 | 2 | 1 | 1 | 1 | 3 | 1 | 288 | 1 | 回声不均/低回声 | 回声不均 | 回声均匀/大小正常/形态大小正常 |
| 10042173 | 3 | 0 | 0.3468 | 0.0877 | 3 | 2 | 2 | 2 | 5 | 1 | 1697 | 5 | 实质回声不均/回声不均/回声欠均/回声欠均匀/低回声 | 实质回声不均/回声不均 | 内部回声细小均匀/未见明显异常回声/大小正常/形态大小正常/未见异常血流 |
| 10127720 | 2 | 1 | 0.4012 | 0.1297 | 3 | 2 | 2 | 2 | 3 | 1 | 722 | 2 | 实质回声不均/回声不均/回声欠均/回声欠均匀/低回声 | 实质回声不均/回声不均 | 大小正常/形态大小正常/未见异常血流 |
| 10014141 | 2 | 1 | 0.4111 | 0.1466 | 5 | 2 | 2 | 2 | 3 | 1 | 1133 | 3 | 实质回声不均/回声不均/回声欠均/回声欠均匀/低回声 | 实质回声不均/回声不均 | 大小正常/形态大小正常/未见异常血流 |
| 10132146 | 3 | 0 | 0.4490 | 0.0667 | 3 | 2 | 2 | 2 | 3 | 1 | 661 | 2 | 实质回声不均/回声不均/回声欠均/回声欠均匀/低回声 | 实质回声不均/回声不均 | 未见明显异常回声/大小正常/形态大小正常 |
| 10157441 | 2 | 1 | 0.4550 | 0.1525 | 3 | 2 | 2 | 2 | 4 | 1 | 877 | 3 | 实质回声不均/回声不均/回声欠均/回声欠均匀/低回声 | 实质回声不均/回声不均 | 内部回声细小均匀/未见明显异常回声/大小正常/形态大小正常 |
| 10064537 | 2 | 1 | 0.4720 | 0.0624 | 5 | 2 | 2 | 2 | 6 | 1 | 2686 | 6 | 实质回声不均/回声不均/回声欠均/回声欠均匀/低回声 | 实质回声不均/回声不均 | 回声均匀/实质回声均匀/未见明显异常回声/大小正常/形态大小正常/未见异常血流 |
| 10131359 | 2 | 1 | 0.4810 | 0.0479 | 3 | 2 | 2 | 2 | 2 | 1 | 1001 | 3 | 实质回声不均/回声不均/回声欠均/回声欠均匀/低回声 | 实质回声不均/回声不均 | 大小正常/形态大小正常 |
| 10064351 | 2 | 1 | 0.5017 | 0.1142 | 3 | 2 | 2 | 2 | 4 | 1 | 1531 | 4 | 实质回声不均/回声不均/回声欠均/回声欠均匀/低回声 | 实质回声不均/回声不均 | 未见明显异常回声/大小正常/形态大小正常/未见异常血流 |
| 10151451 | 2 | 1 | 0.5044 | 0.1095 | 3 | 2 | 2 | 2 | 4 | 1 | 745 | 2 | 实质回声不均/回声不均/回声欠均/回声欠均匀/低回声 | 实质回声不均/回声不均 | 未见明显异常回声/大小正常/形态大小正常/未见异常血流 |
| 10138528 | 2 | 1 | 0.5489 | 0.1847 | 3 | 2 | 2 | 2 | 0 | 0 | 288 | 1 | 弥漫性/回声不均/低回声 | 弥漫性/回声不均 |  |
| 10067220 | 2 | 1 | 0.5574 | 0.1213 | 3 | 2 | 2 | 2 | 5 | 1 | 752 | 3 | 实质回声不均/回声不均/回声欠均/回声欠均匀/低回声 | 实质回声不均/回声不均 | 回声均匀/未见明显异常回声/大小正常/形态大小正常/未见异常血流 |

## Decision Metrics

| stable_fn_patients | stable_tp_patients | stable_fn_mean_first256_diffuse | stable_tp_mean_first256_diffuse | stable_fn_no_diffuse_rate | stable_fn_exposed_positive_rate | stable_fn_mean_full_diffuse |
| --- | --- | --- | --- | --- | --- | --- |
| 19.0000 | 28.0000 | 1.4737 | 1.6071 | 0.1053 | 0.9474 | 1.4737 |

## Decision

`EVIDENCE_EXPOSED_BUT_NOT_USED`.

Next route: stop report-order changes and run analysis-first text representation / fusion contribution audits.

Test reporting-only positive rows generated: 126. They were not used for the decision.
