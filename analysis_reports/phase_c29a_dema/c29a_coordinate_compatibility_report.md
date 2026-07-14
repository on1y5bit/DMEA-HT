# C29-A Coordinate Compatibility Report

- Global compatibility requires validation linear CKA >= 0.70 and patient-distance Spearman >= 0.65.
- kNN Jaccard and train-fitted orthogonal Procrustes validation error are supporting diagnostics and do not independently fail a stage.

| stage | seed pair | CKA | distance Spearman | kNN Jaccard | Procrustes val relative error | compatible |
|---|---|---:|---:|---:|---:|---|
| S2_pre_projection | 0_vs_42 | 0.769360 | 0.764295 | 0.326694 | 0.644668 | True |
| S4_patient_state | 0_vs_42 | 0.814387 | 0.766339 | 0.284348 | 0.705525 | True |
| S2_pre_projection | 0_vs_3407 | 0.657791 | 0.583537 | 0.278505 | 0.715456 | False |
| S4_patient_state | 0_vs_3407 | 0.727710 | 0.631226 | 0.243385 | 0.772376 | False |
| S2_pre_projection | 42_vs_3407 | 0.773795 | 0.726269 | 0.405513 | 0.540250 | True |
| S4_patient_state | 42_vs_3407 | 0.782151 | 0.696259 | 0.337720 | 0.583927 | True |
- S2_pre_projection all-pair global compatibility: `False`
- S4_patient_state all-pair global compatibility: `False`
