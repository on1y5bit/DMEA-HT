# Phase C3 Prediction Shortcut Residual Audit

This is an audit-only analysis. Shortcut fields are never fed into the classifier.

Fields: selected_n_visits, used_images, image_padding_count, has_bio, bio_missing_count, report_length.

| model_id | split | max abs Spearman | linear R2 | shortcut-only label AUC audit-only |
| --- | --- | ---: | ---: | ---: |
| c12_report_filter_stress | test | 0.1036 | 0.0286 | 0.3746 |
| c12_report_filter_stress | val | 0.0946 | 0.0373 | 0.4918 |

Interpretation:

- Chance-level shortcut-only label AUC supports that selected structural fields alone do not recover labels.
- Prediction-shortcut Spearman and linear R2 measure residual association in model outputs.
- High residual association should trigger a pilot, not immediate formal promotion.
