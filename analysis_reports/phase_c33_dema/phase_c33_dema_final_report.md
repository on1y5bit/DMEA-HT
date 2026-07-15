# DEMA-HT Phase C33-JERA Final Report

- Decision: `DEMA_C33_POSITIVE_DAMAGE`.
- Validation AUC mean/std: `0.8711332428 +/- 0.0141305174`.
- Mean Validation gain versus C27: `-0.0111664403`.
- Mean Validation gain versus C32: `-0.0031688547`.
- Reporting-only Test AUC mean/std: `0.8471277400 +/- 0.0241867146`.
- Aggregate C17 TP->C33 FN / FN->C33 TP: `23`/`18`.
- Aggregate C27->C33 repaired/introduced pairs: `98`/`172`.
- Parameter health rows passed: `16/16`.
- Shortcut-only label AUC max: `0.2833861476`.
- Deployment checkpoint: `none`.
- Reporting-only results did not alter architecture, checkpoints, threshold, promotion, or deployment seed.
- Deployment contract remains one checkpoint, one model, one forward, with no prediction combination.
