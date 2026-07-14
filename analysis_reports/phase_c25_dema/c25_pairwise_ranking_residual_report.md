# C25 Pairwise-Ranking Residual

C25 freezes the complete C17 predictor and trains only the unchanged C23 local residual head from detached `mea_mechanism_state`. Its primary objective ranks every positive-negative pair in each mixed training batch; no pointwise classification loss is used.

- formal seeds: `0, 42, 3407`
- checkpoint selection: validation AUC only
- test role: reporting-only after validation selection
- C17 validation AUC mean/std: `0.8696242644 +/- 0.0074797246`
- C25 validation AUC mean/std: `0.8703787536 +/- 0.0073600413`
- C25 minus C17 mean: `+0.0007544892`
- seeds above C17: `2/3`; worst seed delta: `+0.0000000000`
- mixed-class batch coverage gate: `False`; minimum=`0.8668138337`
- correct-positive preservation: `False`; aggregate TP->FN=`0`
- correct-negative preservation: `False`; aggregate TN->FP=`0`
- global-shift gate: `True`
- inversion gate: `True`; repaired=`9`; introduced=`4`
- residual health: `True`
- selected-structure shortcut-only AUC: `0.3277501132`; pass=`True`
- decision: `DEMA_C25_RANK_BATCH_COVERAGE_INVALID`
