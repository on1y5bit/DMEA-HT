# Phase C9 False-Positive Data Audit

Phase C9 is analysis-only. No model, data loader, label, split, manifest, or training code was changed.

## Validation False-Positive Patient Summary

- Unique validation FP patients: 37.
- FP patients present across all three formal seeds: 20.
- FP patients with at least one high-confidence FP seed: 9.
- FP patients with morphology/negative-evidence overlap: 19.
- FP patients with txt_negative_label=1: 14.
- FP patients with aggregation-artifact suspicion: 16.

## Flag Summary

| flag | n_patients | fraction_of_fp_patients |
| --- | --- | --- |
| persistent_fp_all_formal_seeds | 20 | 0.5405 |
| high_confidence_fp_seed_count | 9 | 0.2432 |
| morphology_negative_evidence_overlap | 19 | 0.5135 |
| negative_evidence_positive | 14 | 0.3784 |
| weak_negative_conflict | 13 | 0.3514 |
| morphology_only_fp | 12 | 0.3243 |
| long_report_q4 | 9 | 0.2432 |
| multi_visit_q4 | 15 | 0.4054 |
| aggregation_artifact_suspected | 16 | 0.4324 |

## Highest-Priority Patient Cases

| patient_id | n_unique_fp_seeds | mean_pred_prob | max_pred_prob | high_confidence_fp_seed_count | matched_morphology_terms | matched_negative_terms | report_length | selected_n_visits | aggregation_artifact_suspected | audit_priority |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10106168 | 3 | 0.9232 | 0.9266 | 3 | ['回声欠均', '低回声'] | ['回声均匀', '未见异常血流'] | 441.0000 | 2.0000 | 1 | 8 |
| 10084278 | 3 | 0.8686 | 0.8726 | 3 | ['回声不均', '回声欠均', '回声欠均匀', '低回声'] | ['回声均匀'] | 1739.0000 | 5.0000 | 1 | 8 |
| 10027380 | 3 | 0.7613 | 0.8699 | 1 | ['实质回声不均', '回声不均', '回声欠均', '回声欠均匀', '低回声'] | ['回声均匀', '未见异常血流'] | 805.0000 | 3.0000 | 1 | 8 |
| 10034355 | 3 | 0.7893 | 0.8256 | 1 | ['低回声'] | ['实质回声均匀', '回声均匀'] | 700.0000 | 3.0000 | 1 | 8 |
| 10038703 | 3 | 0.7293 | 0.8169 | 1 | ['回声欠均', '回声欠均匀', '低回声'] | ['实质回声均匀', '回声均匀', '未见异常血流'] | 908.0000 | 3.0000 | 1 | 8 |
| 10043013 | 3 | 0.7974 | 0.8124 | 1 | ['实质回声不均', '回声不均', '回声欠均', '回声欠均匀', '低回声'] | ['未见异常血流'] | 374.0000 | 3.0000 | 1 | 6 |
| 10065841 | 3 | 0.7346 | 0.7827 | 0 | ['回声欠均', '回声欠均匀', '低回声'] | ['回声均匀', '未见异常血流'] | 756.0000 | 4.0000 | 1 | 5 |
| 10032546 | 3 | 0.7390 | 0.7824 | 0 | ['回声欠均', '回声欠均匀', '低回声'] | ['回声均匀'] | 344.0000 | 2.0000 | 1 | 5 |
| 10111232 | 3 | 0.7367 | 0.7731 | 0 | ['回声不均', '回声欠均', '回声欠均匀', '低回声'] | ['实质回声均匀', '回声均匀', '未见异常血流'] | 2507.0000 | 6.0000 | 1 | 5 |
| 10051496 | 3 | 0.6433 | 0.7321 | 0 | ['低回声'] | ['回声均匀', '未见异常血流'] | 609.0000 | 3.0000 | 1 | 5 |
| 10005075 | 3 | 0.6013 | 0.6953 | 0 | ['实质回声不均', '回声不均', '回声欠均', '回声欠均匀', '低回声'] | ['回声均匀', '未见异常血流'] | 1724.0000 | 6.0000 | 1 | 5 |
| 10019805 | 2 | 0.5215 | 0.5368 | 0 | ['低回声'] | ['回声均匀'] | 2502.0000 | 5.0000 | 1 | 5 |
| 10009149 | 3 | 0.7969 | 0.8634 | 1 | ['实质回声不均', '回声不均', '低回声'] | [] | 103.0000 | 1.0000 | 0 | 4 |
| 10023011 | 3 | 0.7750 | 0.8565 | 1 | [] | ['实质回声均匀', '回声均匀', '未见异常血流'] | 835.0000 | 4.0000 | 0 | 4 |
| 10007340 | 2 | 0.7539 | 0.8028 | 1 | ['实质回声不均', '回声不均'] | [] | 123.0000 | 1.0000 | 0 | 4 |

## Interpretation

- This audit does not prove shortcut causality and does not use audit-only fields as classifier inputs.
- Repeated high-confidence false positives across seeds are stronger data-audit targets than single-seed false positives.
- Morphology and negative-evidence overlap is a likely source of misleading patient-level report aggregation.
- Long-report or multi-visit concentration should be treated as a report-construction and case-review signal before changing the model.

## Recommendation

`DATA_CONSTRUCTION_AUDIT_BEFORE_MODEL_CHANGE`.

Before any model or data-construction change, manually review the high-priority patient cases and verify whether positive morphology terms are historical, negated, contradicted, or mixed with later negative evidence.
