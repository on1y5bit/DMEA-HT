# C24 Seed Stability

- formal seeds: `0, 42, 3407`
- checkpoint selection: validation AUC only
- test role: reporting-only after validation selection
- C17 validation AUC mean/std: `0.8696242644 +/- 0.0074797246`
- C24 validation AUC mean/std: `0.8697751622 +/- 0.0070179665`
- C24 minus C17 mean: `+0.0001508978`
- C23 validation AUC mean/std: `0.8703787536 +/- 0.0070761274`
- C24 minus C23 mean: `-0.0006035914`
- decision-side preservation: `True`; violations=`0`
- positive preservation: `False`
- negative preservation: `True`
- high-confidence protection: `True`
- inversion gate: `False`
- residual health: `False`
- selected-structure shortcut-only AUC: `0.3277501132`; pass=`True`
- decision: `DEMA_C24_POSITIVE_SUPPRESSION`
