# Phase C14-E Matching And Top-K Responsibility

Hard positives: `36`; hard negatives: `43`.
Matched hard positives: `11`; matched hard negatives: `4`.

Top-k metrics keep pair coverage, patient-side incidence, and unique-pair responsibility as separate denominators.

| scope | k | inversion_group | top_patient_count | unique_pair_coverage_numerator | unique_pair_coverage_denominator | unique_pair_coverage | patient_side_incidence_numerator | patient_side_incidence_denominator | patient_side_incidence_share | unique_pair_responsibility_numerator | unique_pair_responsibility_denominator | unique_pair_responsibility |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | 5 | all_inversions | 5 | 187 | 380 | 0.4921052631578947 | 525 | 1770 | 0.2966101694915254 | 187 | 380 | 0.4921052631578947 |
| all | 5 | all_seed_inversion | 5 | 151 | 215 | 0.7023255813953488 | 471 | 1290 | 0.36511627906976746 | 151 | 215 | 0.7023255813953488 |
| all | 5 | majority_seed_inversion | 5 | 18 | 75 | 0.24 | 36 | 300 | 0.12 | 18 | 75 | 0.24 |
| all | 5 | single_seed_inversion | 5 | 18 | 90 | 0.2 | 18 | 180 | 0.1 | 18 | 90 | 0.2 |
| all | 10 | all_inversions | 10 | 292 | 380 | 0.7684210526315789 | 799 | 1770 | 0.4514124293785311 | 292 | 380 | 0.7684210526315789 |
| all | 10 | all_seed_inversion | 10 | 198 | 215 | 0.9209302325581395 | 657 | 1290 | 0.5093023255813953 | 198 | 215 | 0.9209302325581395 |
| all | 10 | majority_seed_inversion | 10 | 48 | 75 | 0.64 | 96 | 300 | 0.32 | 48 | 75 | 0.64 |
| all | 10 | single_seed_inversion | 10 | 46 | 90 | 0.5111111111111111 | 46 | 180 | 0.25555555555555554 | 46 | 90 | 0.5111111111111111 |
| all | 20 | all_inversions | 20 | 376 | 380 | 0.9894736842105263 | 1173 | 1770 | 0.6627118644067796 | 376 | 380 | 0.9894736842105263 |
| all | 20 | all_seed_inversion | 20 | 215 | 215 | 1.0 | 921 | 1290 | 0.713953488372093 | 215 | 215 | 1.0 |
| all | 20 | majority_seed_inversion | 20 | 75 | 75 | 1.0 | 166 | 300 | 0.5533333333333333 | 75 | 75 | 1.0 |
| all | 20 | single_seed_inversion | 20 | 86 | 90 | 0.9555555555555556 | 86 | 180 | 0.4777777777777778 | 86 | 90 | 0.9555555555555556 |

## Matching Balance

| label | role | variable | hard_patients | available_controls | matched_hard_patients | unmatched_hard_patients | smd_before | abs_smd_before | smd_after | abs_smd_after | balanced_after |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | positive | report_length | 36 | 11 | 11 | 25 | 0.7450300958037966 | 0.7450300958037966 | 1.1145748339802288 | 1.1145748339802288 | 0 |
| 1 | positive | selected_n_visits | 36 | 11 | 11 | 25 | 0.7983364469541514 | 0.7983364469541514 | 0.926084733220795 | 0.926084733220795 | 0 |
| 1 | positive | used_images | 36 | 11 | 11 | 25 | 0.7983364469541514 | 0.7983364469541514 | 0.926084733220795 | 0.926084733220795 | 0 |
| 1 | positive | image_padding_count | 36 | 11 | 11 | 25 | 0.5434510005764057 | 0.5434510005764057 | 0.5140725757204156 | 0.5140725757204156 | 0 |
| 1 | positive | has_bio | 36 | 11 | 11 | 25 | NA | NA | NA | NA | 0 |
| 1 | positive | bio_missing_count | 36 | 11 | 11 | 25 | 0.4979235152947687 | 0.4979235152947687 | 0.635641726163728 | 0.635641726163728 | 0 |
| 0 | negative | report_length | 43 | 4 | 4 | 39 | 1.1490812192854256 | 1.1490812192854256 | 0.5992491260932534 | 0.5992491260932534 | 0 |
| 0 | negative | selected_n_visits | 43 | 4 | 4 | 39 | 1.494793591606976 | 1.494793591606976 | 1.1078234188139946 | 1.1078234188139946 | 0 |
| 0 | negative | used_images | 43 | 4 | 4 | 39 | 1.494793591606976 | 1.494793591606976 | 1.1078234188139946 | 1.1078234188139946 | 0 |
| 0 | negative | image_padding_count | 43 | 4 | 4 | 39 | 0.4315292692969017 | 0.4315292692969017 | -0.46291004988627577 | 0.46291004988627577 | 0 |
| 0 | negative | has_bio | 43 | 4 | 4 | 39 | NA | NA | NA | NA | 0 |
| 0 | negative | bio_missing_count | 43 | 4 | 4 | 39 | -0.38276837370265687 | 0.38276837370265687 | -0.7071067811865475 | 0.7071067811865475 | 0 |
