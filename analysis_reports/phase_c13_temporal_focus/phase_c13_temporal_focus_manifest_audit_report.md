# Phase C13 Temporal-Focus Manifest Audit

C13 is a data-construction pilot that moves thyroid-relevant latest and historical clauses before the full report text.

## Input And Output

- Input rows: 780.
- Output rows: 780.
- Max focus prefix chars: 220.
- Invariance issues: 0.

## Split/Label Counts

- Input: `{"test": {"0": 42, "1": 42}, "train": {"0": 301, "1": 301}, "val": {"0": 47, "1": 47}}`.
- Output: `{"test": {"0": 42, "1": 42}, "train": {"0": 301, "1": 301}, "val": {"0": 47, "1": 47}}`.

## First-256 Evidence Exposure Summary

| split | label | n | mean_prefix_chars | n_with_prefix | mean_first256_morphology_before | mean_first256_morphology_after | mean_first256_diffuse_before | mean_first256_diffuse_after | txt_morphology_changed_rate | image_weak_changed_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test | 0 | 42 | 170.8095 | 40 | 1.6190 | 1.6190 | 0.4762 | 0.6667 | 0.0000 | 0.0000 |
| test | 1 | 42 | 179.9524 | 42 | 2.4286 | 2.6190 | 1.0714 | 1.4286 | 0.0000 | 0.0000 |
| train | 0 | 301 | 192.8771 | 297 | 1.7508 | 1.9402 | 0.5714 | 0.9136 | 0.0000 | 0.0000 |
| train | 1 | 301 | 193.8073 | 300 | 2.2159 | 2.5548 | 0.8272 | 1.2924 | 0.0000 | 0.0000 |
| val | 0 | 47 | 186.7447 | 45 | 1.7021 | 1.6596 | 0.5319 | 0.6383 | 0.0000 | 0.0000 |
| val | 1 | 47 | 192.6170 | 47 | 2.3191 | 2.8298 | 0.7447 | 1.5532 | 0.0000 | 0.0000 |

## Validation Positive Focus Check

| split | label | n_positive | n_with_prefix | mean_first256_morphology_before | mean_first256_morphology_after | mean_first256_diffuse_before | mean_first256_diffuse_after | n_txt_morphology_changed | n_image_weak_changed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| val | 1 | 47 | 47 | 2.3191 | 2.8298 | 0.7447 | 1.5532 | 0 | 0 |

## Highest Prefix Patients

| patient_id | split | label | focus_prefix_chars | n_latest_focus_clauses | n_history_focus_clauses | first256_morphology_before | first256_morphology_after | first256_diffuse_before | first256_diffuse_after | changed_txt_morphology_label |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10084278 | val | 0 | 220 | 1 | 4 | 6 | 6 | 3 | 3 | 0 |
| 10137142 | test | 1 | 220 | 1 | 2 | 3 | 5 | 0 | 2 | 0 |
| 10078325 | train | 0 | 220 | 1 | 3 | 5 | 5 | 2 | 2 | 0 |
| 10090936 | train | 0 | 220 | 1 | 1 | 5 | 5 | 2 | 2 | 0 |
| 10066465 | train | 1 | 220 | 1 | 1 | 3 | 5 | 0 | 2 | 0 |
| 10089509 | train | 1 | 220 | 1 | 1 | 3 | 5 | 0 | 2 | 0 |
| 10082303 | train | 0 | 220 | 1 | 3 | 4 | 5 | 1 | 2 | 0 |
| 10023866 | train | 1 | 220 | 1 | 3 | 3 | 5 | 0 | 2 | 0 |
| 10110784 | train | 1 | 220 | 1 | 4 | 3 | 5 | 0 | 2 | 0 |
| 10099937 | train | 1 | 220 | 1 | 3 | 3 | 5 | 0 | 2 | 0 |
| 10124560 | train | 0 | 220 | 1 | 3 | 2 | 5 | 1 | 2 | 0 |
| 10005773 | train | 1 | 220 | 1 | 1 | 3 | 5 | 0 | 2 | 0 |
| 10033286 | train | 1 | 220 | 1 | 4 | 2 | 5 | 0 | 2 | 0 |
| 10053940 | train | 1 | 220 | 1 | 1 | 3 | 5 | 0 | 2 | 0 |
| 10072667 | train | 1 | 220 | 1 | 5 | 2 | 5 | 2 | 2 | 0 |

## Interpretation

- The C13 pilot uses report text only, not labels, predictions, or test-selected information.
- Patient IDs, labels, splits, image paths, and bio values must remain invariant.
- The intended mechanism is to reduce truncation loss from long reports under `text_max_length=256`.
- Shortcut and audit fields remain outside the classifier.

## Recommendation

`ALLOW_C13_SINGLE_SEED_TEMPORAL_FOCUS_PILOT`.
