# Phase C12 Report-Construction Pilot Manifest Audit

C12 builds a deterministic report-filter pilot manifest before any model or architecture change.

## Input And Output

- Input rows: 780.
- Output rows: 780.
- Filter mode: `combined_low_risk`.
- Invariance issues: 0.

## Split/Label Counts

- Input: `{"test": {"0": 42, "1": 42}, "train": {"0": 301, "1": 301}, "val": {"0": 47, "1": 47}}`.
- Output: `{"test": {"0": 42, "1": 42}, "train": {"0": 301, "1": 301}, "val": {"0": 47, "1": 47}}`.

## Report Length And Label Change Summary

| split | label | n | n_filtered | filtered_rate | mean_original_report_length | mean_filtered_report_length | mean_report_length_delta | median_report_length_delta | morphology_label_changed_rate | negative_label_changed_rate | image_weak_label_changed_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test | 0 | 42 | 5 | 0.1190 | 365.3095 | 350.4048 | -14.9048 | -2.0000 | 0.0714 | 0.0000 | 0.0714 |
| test | 1 | 42 | 3 | 0.0714 | 357.8571 | 347.3810 | -10.4762 | -2.0000 | 0.0238 | 0.0000 | 0.0238 |
| train | 0 | 301 | 16 | 0.0532 | 552.9934 | 543.6944 | -9.2990 | -3.0000 | 0.0266 | 0.0000 | 0.0266 |
| train | 1 | 301 | 16 | 0.0532 | 547.7508 | 538.3588 | -9.3920 | -3.0000 | 0.0166 | 0.0000 | 0.0166 |
| val | 0 | 47 | 4 | 0.0851 | 571.5957 | 557.1915 | -14.4043 | -3.0000 | 0.0426 | 0.0000 | 0.0426 |
| val | 1 | 47 | 0 | 0.0000 | 525.9574 | 522.2766 | -3.6809 | -2.0000 | 0.0000 | 0.0000 | 0.0000 |

## Text Evidence Label Changes

| field | split | label | n_changed | changed_rate |
| --- | --- | --- | --- | --- |
| txt_morphology_label | test | 0 | 3 | 0.0714 |
| txt_morphology_label | test | 1 | 1 | 0.0238 |
| txt_morphology_label | train | 0 | 8 | 0.0266 |
| txt_morphology_label | train | 1 | 5 | 0.0166 |
| txt_morphology_label | val | 0 | 2 | 0.0426 |
| txt_morphology_label | val | 1 | 0 | 0.0000 |
| txt_negative_label | test | 0 | 0 | 0.0000 |
| txt_negative_label | test | 1 | 0 | 0.0000 |
| txt_negative_label | train | 0 | 0 | 0.0000 |
| txt_negative_label | train | 1 | 0 | 0.0000 |
| txt_negative_label | val | 0 | 0 | 0.0000 |
| txt_negative_label | val | 1 | 0 | 0.0000 |
| txt_uncertain_label | test | 0 | 0 | 0.0000 |
| txt_uncertain_label | test | 1 | 0 | 0.0000 |
| txt_uncertain_label | train | 0 | 0 | 0.0000 |
| txt_uncertain_label | train | 1 | 0 | 0.0000 |
| txt_uncertain_label | val | 0 | 0 | 0.0000 |
| txt_uncertain_label | val | 1 | 0 | 0.0000 |
| txt_diag_hint_label | test | 0 | 0 | 0.0000 |
| txt_diag_hint_label | test | 1 | 0 | 0.0000 |
| txt_diag_hint_label | train | 0 | 0 | 0.0000 |
| txt_diag_hint_label | train | 1 | 0 | 0.0000 |
| txt_diag_hint_label | val | 0 | 0 | 0.0000 |
| txt_diag_hint_label | val | 1 | 0 | 0.0000 |
| image_morphology_weak_label | test | 0 | 3 | 0.0714 |
| image_morphology_weak_label | test | 1 | 1 | 0.0238 |
| image_morphology_weak_label | train | 0 | 8 | 0.0266 |
| image_morphology_weak_label | train | 1 | 5 | 0.0166 |
| image_morphology_weak_label | val | 0 | 2 | 0.0426 |
| image_morphology_weak_label | val | 1 | 0 | 0.0000 |
| discordance_state_label | test | 0 | 0 | 0.0000 |
| discordance_state_label | test | 1 | 0 | 0.0000 |
| discordance_state_label | train | 0 | 0 | 0.0000 |
| discordance_state_label | train | 1 | 0 | 0.0000 |
| discordance_state_label | val | 0 | 0 | 0.0000 |
| discordance_state_label | val | 1 | 0 | 0.0000 |

## Validation Positive Preservation Risk

| split | label | n_positive | n_filtered | filtered_positive_rate | n_txt_morphology_changed | txt_morphology_changed_rate | n_image_weak_changed | image_weak_changed_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| val | 1 | 47 | 0 | 0.0000 | 0 | 0.0000 | 0 | 0.0000 |

## Most Changed Patients

| patient_id | split | label | original_report_length | filtered_report_length | report_length_delta | n_dropped_clauses | latest_diffuse_ht_like | changed_txt_morphology_label | changed_image_morphology_weak_label |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10033988 | train | 0 | 858 | 507 | -351 | 6 | 0 | 0 | 0 |
| 10069437 | train | 1 | 802 | 512 | -290 | 3 | 0 | 0 | 0 |
| 10011560 | train | 0 | 978 | 751 | -227 | 3 | 0 | 0 | 0 |
| 10045094 | train | 1 | 1114 | 926 | -188 | 3 | 0 | 0 | 0 |
| 10120291 | test | 1 | 263 | 122 | -141 | 3 | 0 | 0 | 0 |
| 10068113 | train | 1 | 341 | 205 | -136 | 3 | 0 | 0 | 0 |
| 10015985 | test | 1 | 836 | 725 | -111 | 3 | 0 | 0 | 0 |
| 10006930 | test | 0 | 496 | 289 | -207 | 2 | 0 | 0 | 0 |
| 10024115 | train | 1 | 413 | 230 | -183 | 2 | 0 | 1 | 1 |
| 10130640 | val | 0 | 333 | 158 | -175 | 2 | 0 | 1 | 1 |
| 10032546 | val | 0 | 344 | 171 | -173 | 2 | 0 | 0 | 0 |
| 10039455 | train | 0 | 855 | 702 | -153 | 2 | 0 | 0 | 0 |
| 10016111 | train | 0 | 369 | 218 | -151 | 2 | 0 | 1 | 1 |
| 10079723 | train | 1 | 136 | 0 | -136 | 2 | 0 | 1 | 1 |
| 10102515 | test | 0 | 349 | 222 | -127 | 2 | 0 | 1 | 1 |

## Interpretation

- Patient IDs, labels, splits, image paths, and bio values must remain invariant.
- The report filter uses report text only, not labels, predictions, or test-selected information.
- Shortcut and audit fields remain outside the classifier.

## Recommendation

`ALLOW_C12_SINGLE_SEED_TRAINING_PILOT`.
