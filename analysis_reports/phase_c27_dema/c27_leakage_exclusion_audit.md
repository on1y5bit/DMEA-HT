# C27 Leakage Exclusion Audit

- patient-level split and label counts preserved: `True`
- cross-patient image leakage count: `0`
- selected visit grouping matches C13: `True`
- visit reports match exact patient-date source rows: `True`
- dated bio matches exact patient-date source rows: `True`
- test used for reconstruction design: `False`
- labels, patient IDs, absolute dates, counts, and audit-only shortcuts are absent from the predictor input.
- validation AUC is the only checkpoint and route-promotion metric; test remains reporting-only.
