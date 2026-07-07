# Phase C10 False-Positive Report Source Audit

Phase C10 is analysis-only. No model, data loader, label, split, manifest, or training code was changed.

## Patient-Level Source Summary

- FP patients audited: 37.
- Patients with thyroid positive/negative overlap: 17.
- Patients with benign/nodular mimic signal: 8.
- Patients whose latest visit has negative thyroid cues: 24.
- Patients with diffuse HT-like signal anywhere: 26.

## Flag Summary

| flag | n_patients | fraction_of_fp_patients |
| --- | --- | --- |
| any_thyroid_morphology_hit | 27 | 0.7297 |
| any_non_thyroid_morphology_hit | 6 | 0.1622 |
| any_thyroid_positive_negative_overlap | 17 | 0.4595 |
| any_benign_nodule_mimic | 8 | 0.2162 |
| any_diffuse_ht_like_signal | 26 | 0.7027 |
| latest_visit_has_thyroid_morphology_hit | 25 | 0.6757 |
| latest_visit_has_negative_thyroid_cue | 24 | 0.6486 |
| early_positive_latest_negative_conflict | 12 | 0.3243 |
| focus:thyroid_positive_negative_overlap | 17 | 0.4595 |
| focus:case_level_manual_review | 13 | 0.3514 |
| focus:historical_positive_latest_negative_conflict | 12 | 0.3243 |
| focus:benign_nodule_mimic | 8 | 0.2162 |
| focus:non_thyroid_morphology_source | 3 | 0.0811 |

## Highest-Priority Patient Source Review

| patient_id | n_visits_parsed | max_pred_prob | n_unique_fp_seeds | any_thyroid_positive_negative_overlap | any_benign_nodule_mimic | latest_visit_has_negative_thyroid_cue | early_positive_latest_negative_conflict | recommended_manual_review_focus | source_audit_priority |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10084278 | 5 | 0.8726 | 3 | 1 | 0 | 1 | 1 | thyroid_positive_negative_overlap;historical_positive_latest_negative_conflict | 9 |
| 10038703 | 3 | 0.8169 | 3 | 1 | 1 | 1 | 0 | thyroid_positive_negative_overlap;benign_nodule_mimic | 9 |
| 10065841 | 4 | 0.7827 | 3 | 1 | 1 | 1 | 1 | thyroid_positive_negative_overlap;benign_nodule_mimic;historical_positive_latest_negative_conflict | 8 |
| 10027380 | 3 | 0.8699 | 3 | 1 | 0 | 0 | 0 | thyroid_positive_negative_overlap | 7 |
| 10043013 | 3 | 0.8124 | 3 | 1 | 0 | 0 | 0 | thyroid_positive_negative_overlap | 7 |
| 10034355 | 3 | 0.8256 | 3 | 0 | 0 | 1 | 0 | non_thyroid_morphology_source | 6 |
| 10111232 | 6 | 0.7731 | 3 | 1 | 0 | 1 | 1 | thyroid_positive_negative_overlap;historical_positive_latest_negative_conflict | 6 |
| 10048335 | 1 | 0.7296 | 3 | 1 | 0 | 1 | 1 | thyroid_positive_negative_overlap;historical_positive_latest_negative_conflict | 6 |
| 10013708 | 2 | 0.6650 | 3 | 1 | 0 | 1 | 1 | thyroid_positive_negative_overlap;historical_positive_latest_negative_conflict | 6 |
| 10051496 | 3 | 0.7321 | 3 | 0 | 1 | 1 | 0 | benign_nodule_mimic;non_thyroid_morphology_source | 5 |
| 10030315 | 1 | 0.6408 | 1 | 1 | 0 | 1 | 1 | thyroid_positive_negative_overlap;historical_positive_latest_negative_conflict | 5 |
| 10025710 | 4 | 0.6405 | 1 | 1 | 0 | 1 | 1 | thyroid_positive_negative_overlap;historical_positive_latest_negative_conflict | 5 |
| 10124320 | 2 | 0.6225 | 1 | 1 | 0 | 1 | 1 | thyroid_positive_negative_overlap;historical_positive_latest_negative_conflict | 5 |
| 10007452 | 1 | 0.6043 | 1 | 1 | 0 | 1 | 1 | thyroid_positive_negative_overlap;historical_positive_latest_negative_conflict | 5 |
| 10066099 | 3 | 0.6005 | 1 | 1 | 0 | 1 | 1 | thyroid_positive_negative_overlap;historical_positive_latest_negative_conflict | 5 |
| 10074227 | 2 | 0.5817 | 1 | 1 | 0 | 1 | 1 | thyroid_positive_negative_overlap;historical_positive_latest_negative_conflict | 5 |
| 10019805 | 5 | 0.5368 | 2 | 1 | 1 | 1 | 0 | thyroid_positive_negative_overlap;benign_nodule_mimic | 5 |
| 10092491 | 1 | 0.5221 | 1 | 1 | 0 | 1 | 1 | thyroid_positive_negative_overlap;historical_positive_latest_negative_conflict | 5 |
| 10106168 | 2 | 0.9266 | 3 | 0 | 0 | 0 | 0 | case_level_manual_review | 4 |
| 10009149 | 1 | 0.8634 | 3 | 0 | 0 | 0 | 0 | case_level_manual_review | 4 |

## Visit-Level Evidence Snippets

| patient_id | visit_date | morphology_term_hits_in_thyroid_clauses | negative_term_hits_in_thyroid_clauses | ht_like_thyroid_cue_count | benign_nodule_thyroid_cue_count | negative_thyroid_cue_count | thyroid_positive_and_negative_overlap | benign_nodule_mimic_suspected | thyroid_text_preview |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10084278 | 2019-03-29 | 4 | 0 | 4 | 4 | 2 | 1 | 0 | 甲状腺左侧叶切面形态失常,体积增大,表面光滑,包膜完整,内部回声不均,其内可见一个大小约52×42×32mm的肿块图像,边界清楚,形状呈椭圆形,内部为混合回声,分布不均质,暗区中有部分不规则实性回声,底部回声无衰减。右叶切面形态大小正常,表面光滑,包膜完整,内部回声粗糙均匀,可见几个大小不等最大约10×7mm的低回声结节，边界尚清，内部回声欠均匀 |
| 10084278 | 2020-06-05 | 4 | 0 | 4 | 4 | 2 | 1 | 0 | 甲状腺左侧叶切面形态失常,体积增大,表面光滑,包膜完整,内部回声不均,其内可见一个大小约46×37×33mm的肿块图像,边界清楚,形状呈椭圆形,内部为混合回声,分布不均质,暗区中有部分不规则实性回声,底部回声无衰减。右叶切面形态大小正常,表面光滑,包膜完整,内部回声粗糙均匀,可见几个大小不等最大约12×9mm的低回声结节，边界尚清，内部回声欠均匀 |
| 10084278 | 2021-05-14 | 4 | 0 | 4 | 4 | 2 | 1 | 0 | 甲状腺左侧叶切面形态失常,体积增大,表面光滑,包膜完整,内部回声不均,其内可见一个大小约46×36×33mm的肿块图像,边界清楚,形状呈椭圆形,内部为大量密集点状回声,可见缓慢移动。右叶切面形态大小正常,表面光滑,包膜完整,内部回声粗糙欠均匀,可见几个大小不等最大约14×10mm的低回声结节，边界尚清，内部回声欠均匀 |
| 10084278 | 2022-05-14 | 4 | 0 | 4 | 4 | 2 | 1 | 0 | 甲状腺左侧叶切面形态失常,体积增大,表面光滑,包膜完整,内部回声不均,其内可见一个大小约52×35×33mm的肿块图像,边界清楚,形状呈椭圆形,内部为大量密集点状回声,可见缓慢移动。右侧叶切面形态大小正常,表面光滑,包膜完整,内部回声粗糙欠均匀,可见几个大小不等最大约16×12mm的低回声结节，边界尚清，内部回声欠均匀 |
| 10084278 | 2024-04-26 | 4 | 0 | 4 | 5 | 2 | 1 | 0 | 甲状腺左侧叶切面形态失常,体积增大,表面光滑,包膜完整,内部回声不均,其内可见一个大小约54×37×43mm的囊性为主的结节,边界清楚,形状呈椭圆形,内见密集点状回声。右侧叶切面形态大小正常,表面光滑,包膜完整,内部回声粗糙欠均匀,可见几个较大的约16×12mm的低回声结节，边界尚清，内部回声欠均匀 |
| 10027380 | 2019-09-24 | 1 | 1 | 1 | 3 | 1 | 1 | 0 | 甲状腺切面形态饱满，体积不大，包膜完整,表面欠光滑，内部回声分布不均匀，左侧叶见几个低回声结节较大的约12×7mm，边界清边缘有弧形强回声，内回声均匀，右侧叶可见几个低回声、稍强回声结节，较大的约7×6mm，边界清 |
| 10027380 | 2021-10-25 | 3 | 0 | 1 | 4 | 0 | 0 | 0 | 甲状腺消融术后，体积稍大，形态饱满,表面光滑,包膜完整,两侧叶内部回声细小不均匀,可见多个低回声结节，边界清或欠清，内部回声欠均匀，左侧部分结节内可见强回声斑，后方回声无明显改变，较大结节约：左10×8㎜、右8×5㎜ |
| 10027380 | 2024-09-26 | 3 | 0 | 2 | 3 | 0 | 0 | 0 | 甲状腺消融术后：甲状腺大小形态正常，实质回声不均匀，双侧叶见多个低回声结节，大者约12×8mm，边缘规整，形态呈椭圆形，内部回声不均匀，CDFI：结节内可见少许血流信号 |
| 10034355 | 2018-08-08 | 0 | 0 | 0 | 0 | 4 | 0 | 0 | 甲状腺切面形态大小正常,表面光滑,包膜完整,内部回声细小均匀,其内未见明显异常回声 |
| 10034355 | 2021-07-07 | 0 | 0 | 0 | 0 | 4 | 0 | 0 | 甲状腺切面形态大小正常,表面光滑,包膜完整,内部回声细小均匀,其内未见明显异常回声 |
| 10034355 | 2023-06-28 | 0 | 0 | 0 | 0 | 4 | 0 | 0 | 甲状腺切面形态大小正常,表面光滑,包膜完整,内部回声细小均匀,其内未见明显异常回声 |
| 10038703 | 2018-08-22 | 0 | 0 | 0 | 0 | 4 | 0 | 0 | 甲状腺切面形态大小正常,表面光滑,包膜完整,内部回声细小均匀,其内未见明显异常回声 |
| 10038703 | 2019-06-25 | 1 | 0 | 0 | 4 | 2 | 1 | 1 | 甲状腺切面形态大小正常,表面光滑,包膜完整,右侧叶内部回声细小不均匀,可见一个低回声结节，大小约2×2mm，边界清，内部回声尚均匀，后方回声无明显改变。左侧叶及峡部回声细小均匀，内未见明显局限性异常回声 |
| 10038703 | 2022-11-01 | 3 | 0 | 1 | 4 | 2 | 1 | 0 | 甲状腺切面形态大小正常，表面光滑，包膜完整，右侧叶内部回声细小不均匀，可见一个低回声结节，大小约5×3mm，边界清，内部回声欠均匀，后方回声无明显改变。左侧叶及峡部回声细小均匀，内未见明显局限性异常回声 |
| 10043013 | 2022-09-05 | 3 | 0 | 1 | 4 | 2 | 1 | 0 | 甲状腺切面形态大小正常,表面光滑,包膜完整,右侧叶内部回声细小不均匀,可见两个低回声结节，较大约4×3mm，边界清，内部回声欠均匀，后方回声无明显改变。左侧叶及峡部回声细小均匀，内未见明显局限性异常回声 |
| 10043013 | 2023-10-16 | 3 | 0 | 2 | 3 | 1 | 1 | 0 | 甲状腺大小形态正常，实质回声不均匀，双侧叶见几个低回声结节，大者约3×3mm，边缘规整，形态呈椭圆形，内部回声不均匀。峡部未见明显异常回声 |
| 10043013 | 2024-09-18 | 3 | 0 | 2 | 3 | 0 | 0 | 0 | 甲状腺大小形态正常，实质回声不均匀，双侧叶见几个低回声结节，大者约5×4mm，边缘规整，形态呈椭圆形，内部回声不均匀 |
| 10065841 | 2019-03-25 | 1 | 1 | 0 | 6 | 4 | 1 | 1 | 甲状腺切面形态大小正常,表面光滑,包膜完整,内部回声细小均匀,左侧叶可见2个低回声结节，最大约7.4mm×3.6mm，结节边界清晰，内部回声均匀，未见明显血流信号。右侧叶可见一个低回声结节，大小约4.6mm×3.4mm，结节边界清晰，内部回声均匀，未见明显血流信号 |
| 10065841 | 2020-07-08 | 3 | 0 | 1 | 5 | 2 | 1 | 0 | 甲状腺切面形态大小正常,表面光滑,包膜完整,两侧叶内部回声细小不均匀,可见几个低回声结节，较大约：左7×4mm、右5×3㎜，边界清，内部回声欠均匀，后方回声无明显改变，CDFI：结节内及周边未见明显血流信号 |
| 10065841 | 2021-03-29 | 3 | 0 | 1 | 5 | 2 | 1 | 0 | 甲状腺切面形态大小正常,表面光滑,包膜完整,两侧叶内部回声细小不均匀,可见几个低回声、混合性回声结节，较大约：左5×3mm、右6×4㎜，形态规则，边界清，内部回声欠均匀，后方回声无明显改变，CDFI：结节内部及周边未见明显血流信号 |
| 10065841 | 2022-03-23 | 3 | 0 | 1 | 4 | 2 | 1 | 0 | 甲状腺切面形态大小正常,表面光滑,包膜完整,两侧叶内部回声细小不均匀,可见几个低回声结节，大者位于右侧叶，大小约6×4mm，边界清，内部回声欠均匀，后方回声无明显改变 |
| 10111232 | 2018-11-02 | 3 | 0 | 1 | 4 | 2 | 1 | 0 | 甲状腺切面形态大小正常,表面光滑,包膜完整,左、右侧叶内部回声细小不均匀, 可见几个低回声结节，边界清，内部回声欠均匀，后方回声无明显改变，较大者分别为 7 × 5 ㎜、4 × 2 ㎜ |
| 10111232 | 2019-10-25 | 3 | 0 | 1 | 4 | 2 | 1 | 0 | 甲状腺切面形态大小正常,表面光滑,包膜完整,两侧叶内部回声细小不均匀,可见几个低回声结节，较大约：左8×5㎜、右6×3mm，边界清，内部回声欠均匀，后方回声无明显改变，CDFI：部分结节周边可见少许血流信号 |
| 10111232 | 2020-11-26 | 3 | 0 | 1 | 4 | 2 | 1 | 0 | 甲状腺切面形态大小正常,表面光滑,包膜完整,两侧叶内部回声细小不均匀,可见几个低回声结节，较大约：左8×5㎜、右6×3mm，边界清，内部回声欠均匀，后方回声无明显改变，CDFI：部分结节周边可见少许血流信号 |
| 10111232 | 2021-09-12 | 3 | 0 | 1 | 4 | 2 | 1 | 0 | 甲状腺切面形态大小正常,表面光滑,包膜完整,两侧叶内部回声细小不均匀,可见几个低回声结节，较大约8×6mm，边界清，内部回声欠均匀，后方回声无明显改变，CDFI：部分结节周边可见少许血流信号 |
| 10111232 | 2022-05-12 | 3 | 0 | 1 | 4 | 2 | 1 | 0 | 甲状腺切面形态大小正常,表面光滑,包膜完整,两侧叶内部回声细小不均匀,可见几个低回声结节，较大约9×6mm，边界清，内部回声欠均匀，后方回声无明显改变，CDFI：部分结节周边可见少许血流信号 |
| 10111232 | 2023-04-13 | 3 | 0 | 1 | 4 | 2 | 1 | 0 | 甲状腺切面形态大小正常,表面光滑,包膜完整,两侧叶内部回声细小不均匀,可见几个低回声结节，较大约9×7mm，边界清，内部回声欠均匀，后方回声无明显改变，CDFI：部分结节周边可见少许血流信号 |
| 10048335 | 2023-10-23 | 3 | 0 | 1 | 6 | 2 | 1 | 0 | 甲状腺切面形态大小正常,表面光滑,包膜完整,两侧叶内部回声细小不均匀,可见几个低回声结节，较大约8×6mm（左）、13×6mm（右），形态规则，边界清，内部回声欠均匀，后方回声无明显改变，CDFI:结节内部及周边未见血流信号 |

## Interpretation

- Many strict MVP false positives appear tied to mixed longitudinal evidence rather than a simple model-only error.
- Benign thyroid nodule language and low/uneven echo terms can resemble HT morphology while still belonging to label-negative patients.
- Latest-visit negative cues and historical positive cues should be reviewed before changing the classifier.
- Shortcut fields remain audit-only; this report does not justify feeding visit count, image count, or report length into a model.

## Recommendation

`AUDIT_TEMPORAL_EVIDENCE_CONFLICT_BEFORE_TRAINING`.

Before any training pilot, define a report-construction or evidence-filtering hypothesis that can be audited without labels changing and without shortcut variables entering the classifier.
