# DEMA-HT Phase C32-VPA Final Report

- Decision: `DEMA_C32_POSITIVE_DAMAGE`.
- Validation AUC mean/std: `0.8743020975 +/- 0.0171207551`.
- Mean Validation gain versus C27: `-0.0079975856`.
- Reporting-only Test AUC mean/std: `0.8454270597 +/- 0.0258190758`.
- Aggregate C17 TP->C32 FN / FN->C32 TP: `21`/`16`.
- Aggregate C27->C32 repaired/introduced pairs: `53`/`106`.
- Projector health rows passed: `9/9`.
- Shortcut-only label AUC max: `0.2833861476`.
- Deployment checkpoint: `none`.
- Reporting-only results did not alter architecture, checkpoints, threshold, promotion, or deployment seed.
- Deployment contract remains one checkpoint, one model, one forward, with no prediction combination.
