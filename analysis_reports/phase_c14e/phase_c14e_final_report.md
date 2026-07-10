# Phase C14-E Hard Clinical Evidence Audit

C14-E is analysis-only. No model/training code, labels, splits, task, manifest, report construction, images, bio values, or thresholds were changed. Test results were not used.

## Cohorts

- Hard positives: `36`; non-hard positives: `11`.
- Hard negatives: `43`; non-hard negatives: `4`.

## Top-K Responsibility

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

## Matching Quality

- Positive matching coverage: `0.3056`; negative matching coverage: `0.0930`.
- Unmatched hard positives: `25`; unmatched hard negatives: `39`.
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

## Candidate Mechanism Coverage

| candidate_mechanism | hard_patients_explained | hard_patient_count | hard_fraction_explained | matched_controls_with_mechanism | matched_control_count | matched_control_fraction | risk_difference_hard_minus_control | cross_seed_consistency | top_two_patient_dependence | matching_coverage | maps_to_valid_intervention | passes_30pct_generalizability_gate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hard_positive_weak_or_ambiguous_evidence | 30 | 36 | 0.8333333333333334 | 11 | 11 | 1.0 | -0.16666666666666663 | 3 | 0.06666666666666667 | 0.3055555555555556 | HT_SPECIFIC_TEXT_EVIDENCE_AUDIT_OR_REPRESENTATION_PILOT | 0 |
| hard_negative_ht_like_image_mimic | 2 | 43 | 0.046511627906976744 | 2 | 4 | 0.5 | -0.4534883720930233 | 3 | 1.0 | 0.09302325581395349 | IMAGE_MIMIC_ROBUSTNESS_PILOT_DESIGN | 0 |
| label_or_followup_ambiguity | 30 | 43 | 0.6976744186046512 | 4 | 4 | 1.0 | -0.3023255813953488 | 3 | 0.06666666666666667 | 0.09302325581395349 | DATA_AND_LABEL_AUDIT_ONLY | 0 |

Largest observed mechanism: `hard_positive_weak_or_ambiguous_evidence` with hard-subgroup fraction `0.8333`.
Hard-negative label/follow-up ambiguity prevalence: `0.6977`.

## Final Route

`DATA_LIMIT_NO_GENERAL_MODEL_FIX`.

Allowed next-step class: `KEEP_C13_AND_REPORT_LIMITATION`.
Decision basis: Matched-control coverage is below 50% for at least one label subgroup, preventing a broad model mechanism claim.

No route in C14-E automatically authorizes training. C15 remains blocked pending a separate explicit decision. C13 remains the current strict best; no model improvement or AUC 0.90 claim is made.
