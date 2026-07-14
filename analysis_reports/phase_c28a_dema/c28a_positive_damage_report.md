# C28-A Positive-Damage Report

- seed 0: material=`12`; severe=`8`.
  - V1_uniform: available=`12`; material rescues=`2`; TP-to-FN rescues=`2`.
  - V2_recency_only: available=`12`; material rescues=`4`; TP-to-FN rescues=`3`.
  - V3_content_only: available=`12`; material rescues=`1`; TP-to-FN rescues=`1`.
  - V4_latest_only: available=`12`; material rescues=`8`; TP-to-FN rescues=`3`.
  - V5_history_mean_only: available=`10`; material rescues=`1`; TP-to-FN rescues=`1`.
  - conflict_group damage rates: multi_visit_low_conflict 5/14 (0.357); multi_visit_medium_conflict 5/17 (0.294); single_visit 2/16 (0.125).
  - text_evidence_group damage rates: latest_history_mixed_or_uncertain 6/19 (0.316); latest_negative_like_history_positive_like 1/1 (1.000); latest_positive_like_history_negative_like 3/11 (0.273); single_visit 2/16 (0.125).
- seed 42: material=`21`; severe=`19`.
  - V1_uniform: available=`21`; material rescues=`0`; TP-to-FN rescues=`0`.
  - V2_recency_only: available=`21`; material rescues=`1`; TP-to-FN rescues=`0`.
  - V3_content_only: available=`21`; material rescues=`0`; TP-to-FN rescues=`1`.
  - V4_latest_only: available=`21`; material rescues=`10`; TP-to-FN rescues=`5`.
  - V5_history_mean_only: available=`17`; material rescues=`1`; TP-to-FN rescues=`1`.
  - conflict_group damage rates: multi_visit_low_conflict 4/9 (0.444); multi_visit_medium_conflict 13/22 (0.591); single_visit 4/16 (0.250).
  - text_evidence_group damage rates: latest_history_mixed_or_uncertain 11/19 (0.579); latest_negative_like_history_positive_like 1/1 (1.000); latest_positive_like_history_negative_like 5/11 (0.455); single_visit 4/16 (0.250).
- seed 3407: material=`9`; severe=`8`.
  - V1_uniform: available=`9`; material rescues=`0`; TP-to-FN rescues=`0`.
  - V2_recency_only: available=`9`; material rescues=`0`; TP-to-FN rescues=`0`.
  - V3_content_only: available=`9`; material rescues=`0`; TP-to-FN rescues=`0`.
  - V4_latest_only: available=`9`; material rescues=`1`; TP-to-FN rescues=`1`.
  - V5_history_mean_only: available=`6`; material rescues=`1`; TP-to-FN rescues=`1`.
  - conflict_group damage rates: multi_visit_low_conflict 3/9 (0.333); multi_visit_medium_conflict 3/22 (0.136); single_visit 3/16 (0.188).
  - text_evidence_group damage rates: latest_history_mixed_or_uncertain 1/19 (0.053); latest_negative_like_history_positive_like 1/1 (1.000); latest_positive_like_history_negative_like 4/11 (0.364); single_visit 3/16 (0.188).

Only material probability loss or threshold transitions are interpreted; smaller patient-level changes are retained as descriptive evidence.
