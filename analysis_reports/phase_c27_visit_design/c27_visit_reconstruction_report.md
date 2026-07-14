# C27 Visit Reconstruction Report

- decision: `C27_VISIT_RECONSTRUCTION_PASS`
- patients reconstructed: `780`
- selected visits reconstructed: `1932`
- visit-level report coverage: `0.9984472050` (required `>=0.80`)
- multi-visit validation two-block coverage: `1.0000000000` (required `>=0.70`)
- cross-patient image leakage count: `0`
- split/label invariance: `True`
- manifest SHA256: `cc19e7d1088a5df79b937fc8db4196300796a2adbfe2cb49f42be0f99b4a5b9b`
- visit boundaries: C13 selected real visit dates only
- image grouping: original C13 selected image paths grouped by source visit directory
- report blocks: exact patient-date rows from `all_patients.xlsx`
- dated bio: exact patient-date source row only
- missing reports: empty with source reason; patient concatenated report is never copied
- test role: invariance audit only; no reconstruction rule or threshold was selected on test

C27_VISIT_RECONSTRUCTION_PASS
