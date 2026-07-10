# Phase C14-B Reproduction Check

This is a mandatory pre-audit gate. Downstream representation, masking, and occlusion claims are valid only when every required seed passes.

| seed | checkpoint_path | saved_prediction_rows | reproduced_prediction_rows | patient_id_match | label_match | max_abs_prob_diff | mean_abs_prob_diff | reproduction_pass | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | runs/dmea_ht_v2_c13_temporal_focus_stress_seeds/checkpoints/seed_0_best.pt | 94 | 94 | 1 | 1 | 1.1102230246251565e-16 | 2.3400312420623313e-17 | 1 | tokenizer=character-level/text_max_length=256; image_size=224; max_images=28; duplicate_reproduced=False |
| 42 | runs/dmea_ht_v2_c13_temporal_focus_stress_seeds/checkpoints/seed_42_best.pt | 94 | 94 | 1 | 1 | 1.1102230246251565e-16 | 2.2588314197825658e-17 | 1 | tokenizer=character-level/text_max_length=256; image_size=224; max_images=28; duplicate_reproduced=False |
| 3407 | runs/dmea_ht_v2_c13_temporal_focus_stress_seeds/checkpoints/seed_3407_best.pt | 94 | 94 | 1 | 1 | 1.1102230246251565e-16 | 2.048450062057719e-17 | 1 | tokenizer=character-level/text_max_length=256; image_size=224; max_images=28; duplicate_reproduced=False |

Overall reproduction gate: `PASS`.
Required thresholds: max absolute probability difference <= 1e-5 and mean absolute probability difference <= 1e-6, with matching patient IDs and labels.
