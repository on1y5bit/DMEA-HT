# Phase C27 DEMA-HT Final Report

- canonical project: `/home/linruixin/chen/project/DMEA-HT`
- runtime: `/home/linruixin/chen/conda/envs/ma`
- single-model route only; no ensemble or checkpoint averaging
- C26-E status: `C26E_WITHDRAWN_BY_USER`; no ensemble artifact exists
- C26-SM status: `STOP_C26SM_TUNING`
- deployment contract: one checkpoint, one model, one forward
- checkpoint selection and route promotion: validation AUC only; test reporting-only
- visit reconstruction: `C27_VISIT_RECONSTRUCTION_PASS`; report coverage=`0.9984472050`
- C17 validation AUC mean/std: `0.8696242644 +/- 0.0074797246`
- C26-SM validation AUC mean/std: `0.8702278557 +/- 0.0084731523`
- C27 validation AUC mean/std: `0.8822996831 +/- 0.0168958421`
- C27 minus C17 mean: `+0.0126754187`; AUC gate=`False`
- positive preservation: `False`; TP->FN/FN->TP=`17/16`
- inversion gate: `False`; repaired/introduced=`464/380`
- temporal health: `False`
- stability means: CKA=`0.7747494883`, distance Spearman=`0.6979414525`, kNN Jaccard=`0.2884845203`; pass=`False`
- selected-structure shortcut-only AUC/max prediction correlation: `0.2833861476`/`0.1915568467`; pass=`True`
- reporting-only test AUC mean/std: `0.8439153439 +/- 0.0299238834`
- decision: `DEMA_C27_POSITIVE_RECALL_DAMAGE`

KEEP_DEMA_C17_STRICT_BEST
STOP_C27_VTME_TUNING
