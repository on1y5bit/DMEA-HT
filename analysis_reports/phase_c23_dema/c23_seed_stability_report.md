# C23 Seed Stability

- formal seeds: `0, 42, 3407`
- checkpoint selection: validation AUC only
- test role: reporting-only after validation selection
- C17 validation AUC mean/std: `0.8696242644 +/- 0.0074797246`
- C23 validation AUC mean/std: `0.8703787536 +/- 0.0070761274`
- C23 minus C17 mean: `+0.0007544892`
- positive preservation: `False`
- negative preservation: `True`
- high-confidence protection: `True`
- inversion gate: `True`
- residual health: `True`
- selected-structure shortcut-only AUC: `0.3277501132`; pass=`True`
- decision: `DEMA_C23_POSITIVE_SUPPRESSION`
