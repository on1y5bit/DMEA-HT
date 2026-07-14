# C26-SM Single-Model Deployment Contract

Each formal seed is an independent training replicate. Deployment inference loads exactly one C26-SM checkpoint, one model, and performs one forward pass. Checkpoints and predictions are never averaged, voted, stacked, or jointly loaded.

- C26-E status: `C26E_WITHDRAWN_BY_USER`; no ensemble artifact exists
- deployment contract: one checkpoint, one model, one forward
- checkpoint selection: validation AUC only; test reporting-only
- C17 validation AUC mean/std: `0.8696242644 +/- 0.0074797246`
- C22 validation AUC mean/std: `0.8678134903 +/- 0.0090425461`
- C26-SM validation AUC mean/std: `0.8702278557 +/- 0.0084731523`
- C26-SM minus C17 mean: `+0.0006035914`; AUC gate=`False`
- positive preservation: `False`; aggregate TP->FN=`0`
- inversion gate: `False`; repaired/introduced=`45/41`
- mechanism stability means: CKA=`0.4561818549`, distance Spearman=`0.4665281165`, kNN Jaccard=`0.2741969850`; pass=`False`
- relation-gate cross-seed dispersion: `0.0002462321`
- mechanism health: `True`; residual health=`False`
- selected-structure shortcut-only AUC: `0.2829334541`; pass=`True`
- reporting-only test AUC mean/std: `0.8408919123 +/- 0.0016364804`
- decision: `DEMA_C26SM_POSITIVE_SUPPRESSION`
