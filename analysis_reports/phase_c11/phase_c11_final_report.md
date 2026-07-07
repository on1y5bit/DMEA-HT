# Phase C11 Report-Filter Hypothesis Audit

Phase C11 is analysis-only. It audits report-construction hypotheses before any training pilot.

## Validation Cohort

- Validation patients audited: 94.
- Mean-threshold false-positive patients: 24.
- Label-positive patients for positive-preservation risk: 47.

## Hypothesis Summary

| hypothesis | n_flagged_all_val | n_flagged_mean_fp | mean_fp_capture_rate | n_flagged_label1_positive | positive_patient_flag_rate | n_flagged_true_positive | true_positive_flag_rate | n_flagged_false_negative | precision_for_mean_fp_among_flagged | recommendation_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| positive_negative_overlap_review | 58 | 11 | 0.4583 | 35 | 0.7447 | 32 | 0.7442 | 3 | 0.1897 | AUDIT_MORE_HIGH_POSITIVE_RISK |
| latest_negative_suppresses_history | 30 | 7 | 0.2917 | 16 | 0.3404 | 14 | 0.3256 | 2 | 0.2333 | AUDIT_MORE_HIGH_POSITIVE_RISK |
| non_thyroid_morphology_only | 5 | 3 | 0.1250 | 0 | 0.0000 | 0 | 0.0000 | 0 | 0.6000 | CASE_REVIEW_ONLY |
| benign_nodule_without_latest_diffuse | 24 | 11 | 0.4583 | 5 | 0.1064 | 5 | 0.1163 | 0 | 0.4583 | PILOT_ELIGIBLE_LOW_POSITIVE_RISK |
| require_latest_diffuse_ht_like | 20 | 9 | 0.3750 | 4 | 0.0851 | 4 | 0.0930 | 0 | 0.4500 | PILOT_ELIGIBLE_LOW_POSITIVE_RISK |

## Positive-Preservation Risk

| hypothesis | n_positive_flagged | mean_pred_prob_positive_flagged | n_true_positive_flagged | n_false_negative_flagged | example_positive_patient_ids |
| --- | --- | --- | --- | --- | --- |
| latest_negative_suppresses_history | 16 | 0.6770 | 14 | 2 | 10005340,10042173,10045442,10063399,10064351,10064537,10064626,10066912,10067220,10098069 |
| benign_nodule_without_latest_diffuse | 5 | 0.6504 | 5 | 0 | 10012205,10098069,10117252,10119735,10132330 |
| require_latest_diffuse_ht_like | 4 | 0.6794 | 4 | 0 | 10098069,10117252,10119735,10132330 |
| non_thyroid_morphology_only | 0 | NA | 0 | 0 |  |
| positive_negative_overlap_review | 35 | 0.7169 | 32 | 3 | 10005340,10014141,10022640,10024132,10038097,10042173,10045442,10057459,10063399,10064351 |

## Highest-Probability FP Examples

| patient_id | mean_pred_prob | latest_negative_suppresses_history | benign_nodule_without_latest_diffuse | require_latest_diffuse_ht_like | non_thyroid_morphology_only | positive_negative_overlap_review | latest_visit_date | latest_thyroid_preview |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10106168 | 0.9232 | 0 | 0 | 0 | 0 | 0 | 2022-08-08 | 甲状腺切面形态失常,两侧叶呈弥漫性、对称性稍大,峡部稍厚，表面欠光滑,包膜完整,内部回声普遍增粗增强,弥漫性紊乱,右侧叶见大小约9×6mm等回声结节，左侧叶近峡部可见大小约12×5mm的低回声，形态规则，结节边界尚清，内部回声欠均，CDFI：甲状腺右侧叶结节可见环状的血流信号 |
| 10084278 | 0.8686 | 1 | 0 | 0 | 0 | 1 | 2024-04-26 | 甲状腺左侧叶切面形态失常,体积增大,表面光滑,包膜完整,内部回声不均,其内可见一个大小约54×37×43mm的囊性为主的结节,边界清楚,形状呈椭圆形,内见密集点状回声。右侧叶切面形态大小正常,表面光滑,包膜完整,内部回声粗糙欠均匀,可见几个较大的约16×12mm的低回声结节，边界尚清，内部回声欠均匀 |
| 10043013 | 0.7974 | 0 | 0 | 0 | 0 | 1 | 2024-09-18 | 甲状腺大小形态正常，实质回声不均匀，双侧叶见几个低回声结节，大者约5×4mm，边缘规整，形态呈椭圆形，内部回声不均匀 |
| 10009149 | 0.7969 | 0 | 0 | 0 | 0 | 0 | 2024-07-20 | 甲状腺大小形态正常，实质回声不均匀，双侧叶见多个低回声结节，大者约8×4mm，边缘规整，形态呈椭圆形，内部回声不均匀，CDFI：结节内可见少许血流信号 |
| 10034355 | 0.7893 | 0 | 0 | 0 | 1 | 0 | 2023-06-28 | 甲状腺切面形态大小正常,表面光滑,包膜完整,内部回声细小均匀,其内未见明显异常回声 |
| 10023011 | 0.7750 | 0 | 0 | 0 | 0 | 0 | 2024-06-13 | 甲状腺切面形态大小正常,表面光滑,包膜完整,内部回声细小均匀,其内未见明显异常回声 |
| 10027380 | 0.7613 | 0 | 0 | 0 | 0 | 1 | 2024-09-26 | 甲状腺消融术后：甲状腺大小形态正常，实质回声不均匀，双侧叶见多个低回声结节，大者约12×8mm，边缘规整，形态呈椭圆形，内部回声不均匀，CDFI：结节内可见少许血流信号 |
| 10032546 | 0.7390 | 0 | 1 | 1 | 0 | 0 | 2024-05-20 | 甲状腺部分切除术后,剩余双侧叶大小尚可，右侧叶内可见两个低回声结节，大小约5×3mm、4×3mm，形态规则，边界清，内部回声欠均匀，后方回声无明显改变，CDFI:结节内部及周边未见血流信号 |
| 10111232 | 0.7367 | 1 | 1 | 1 | 0 | 1 | 2023-04-13 | 甲状腺切面形态大小正常,表面光滑,包膜完整,两侧叶内部回声细小不均匀,可见几个低回声结节，较大约9×7mm，边界清，内部回声欠均匀，后方回声无明显改变，CDFI：部分结节周边可见少许血流信号 |
| 10065841 | 0.7346 | 1 | 1 | 1 | 0 | 1 | 2022-03-23 | 甲状腺切面形态大小正常,表面光滑,包膜完整,两侧叶内部回声细小不均匀,可见几个低回声结节，大者位于右侧叶，大小约6×4mm，边界清，内部回声欠均匀，后方回声无明显改变 |
| 10038703 | 0.7293 | 1 | 1 | 1 | 0 | 1 | 2022-11-01 | 甲状腺切面形态大小正常，表面光滑，包膜完整，右侧叶内部回声细小不均匀，可见一个低回声结节，大小约5×3mm，边界清，内部回声欠均匀，后方回声无明显改变。左侧叶及峡部回声细小均匀，内未见明显局限性异常回声 |
| 10137578 | 0.7140 | 0 | 0 | 0 | 0 | 0 | 2023-08-08 | 甲状腺大小形态正常，实质回声不均匀，双侧叶可见几个低回声结节，较大约8×6mm，边缘规整，形态呈椭圆形，内部回声不均匀 |
| 10001110 | 0.6962 | 0 | 0 | 0 | 0 | 0 | 2024-09-02 | 甲状腺大小形态正常，实质回声不均匀，双侧叶见多个低回声结节，大者约20×13mm，边缘规整，形态呈椭圆形，内部回声不均匀，CDFI：结节内可见少许血流信号 |
| 10004992 | 0.6826 | 0 | 0 | 0 | 0 | 0 | 2023-11-29 | 甲状腺部分切除术后，形态失常，右侧叶大于左侧叶,两侧叶表面欠光滑,内部回声增粗不均匀,右侧叶内见几个大小2-3mm无回声结节，左叶可见大小约12×8mm混合性回声结节，边界清，内部回声不均匀。CDFI：甲状腺内血流信号略显增多 |
| 10007340 | 0.6558 | 0 | 0 | 0 | 0 | 0 | 2024-09-12 | 甲状腺大小形态正常，实质回声不均匀，双侧叶见多个低无回声结节，左侧大者约3.6×2.3mm，右侧大者约19×17mm，边缘规整，形态呈椭圆形，内部回声不均匀，CDFI：结节内可见少许血流信号 |

## Interpretation

- A hypothesis is not a model change; it is only eligible for a later low-cost pilot if it captures FP cases without flagging many label-positive patients.
- Test data is not used here.
- Shortcut fields remain audit-only and are not candidate model inputs.

## Recommendation

`ALLOW_REPORT_FILTER_PILOT_FOR_LOW_RISK_HYPOTHESIS`.

If a later pilot is allowed, it must be report-construction only, validation-selected, bad-seed/stress-seed checked, and followed by positive-preservation and shortcut residual audits.
