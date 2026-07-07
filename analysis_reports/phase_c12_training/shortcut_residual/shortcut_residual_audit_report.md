# Phase C3 Prediction Shortcut Residual Audit

This is an audit-only analysis. Shortcut fields are never fed into the classifier.

Fields: selected_n_visits, used_images, image_padding_count, has_bio, bio_missing_count, report_length.

| model_id | split | max abs Spearman | linear R2 | shortcut-only label AUC audit-only |
| --- | --- | ---: | ---: | ---: |
| c12_report_filter_pilot | test | 0.3562 | 0.1055 | 0.2883 |
| c12_report_filter_pilot | val | 0.2394 | 0.0821 | 0.3332 |

Interpretation:

- Chance-level shortcut-only label AUC supports that selected structural fields alone do not recover labels.
- Prediction-shortcut Spearman and linear R2 measure residual association in model outputs.
- High residual association should trigger a pilot, not immediate formal promotion.
