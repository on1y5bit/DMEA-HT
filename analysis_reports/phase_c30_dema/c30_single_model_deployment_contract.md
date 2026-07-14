# C30 Single-Model Deployment Contract

- official model name remains `DEMA-HT`
- C17 Positive Preservation remains the strict best before C30
- C27 is the strongest unpromoted new-backbone signal before C30
- C30 changes only the visit-text token representation before frozen text-evidence pooling
- validation AUC is the only checkpoint and promotion metric; test is reporting-only
- one checkpoint, one model, one forward; predictions and checkpoint weights are not combined
- selected epochs: `{0: 1, 42: 1, 3407: 1}`
- C17 validation AUC mean/std: `0.8696242644 +/- 0.0074797246`
- C27 validation AUC mean/std: `0.8822996831 +/- 0.0168958421`
- C30 validation AUC mean/std: `0.8783763392 +/- 0.0165980767`
- C30 minus C17 per seed: `[0.027161611588954138, -0.010864644635581677, 0.00995925758261662]`; mean `+0.0087520748`
- C30 minus C27 per seed: `[-0.004526935264825727, -0.005432322317790783, -0.001810774105930335]`; mean `-0.0039233439`
- positive preservation pass: `False`; C17 TP->FN/FN->TP aggregate `15/17`
- material positive damage C27/C30: `42/40`; reduction `0.0476190476`
- inversion pass: `False`; C27->C30 repaired/introduced `13/39`
- adapter health pass: `True`; near-bound max `0.0000000000`; token cosine min `0.9999975911`; padding delta max `0.0000000000`
- shortcut-only AUC/max prediction correlation: `0.2833861476`/`0.1961086067`; pass `True`
- reporting-only test AUC mean/std: `0.8450491308 +/- 0.0294802750`
- decision: `DEMA_C30_POSITIVE_RECALL_DAMAGE`

Each seed is an independent architecture-stability replicate. Deployment loads one validation-selected checkpoint and executes one model forward pass.
- C30 is not promoted; the C17 strict-best deployment remains unchanged.
