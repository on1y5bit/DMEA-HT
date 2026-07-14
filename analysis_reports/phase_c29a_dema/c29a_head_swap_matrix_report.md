# C29-A Cross-Seed Head-Swap Report

- Rows are representation seeds and columns are applied head seeds.
- Diagonal cells reproduce the official C27 predictor; off-diagonal cells are diagnostics only.
- No cell is an averaged prediction, deployment candidate, or seed-selection rule.

## Classifier-Only Validation AUC

| representation seed | 0 | 42 | 3407 |
|---:|---:|---:|---:|
| 0 | 0.9017655048 | 0.8415572657 | 0.7655047533 |
| 42 | 0.3920325939 | 0.8714350385 | 0.7170665459 |
| 3407 | 0.6727025804 | 0.4599366229 | 0.8736985061 |

## Projection Plus Classifier Validation AUC

| representation seed | 0 | 42 | 3407 |
|---:|---:|---:|---:|
| 0 | 0.9017655048 | 0.6315074694 | 0.7297419647 |
| 42 | 0.5464010865 | 0.8714350385 | 0.7962879131 |
| 3407 | 0.2304210050 | 0.7270258035 | 0.8736985061 |

- maximum classifier diagonal logit error: `9.53674316406e-07`
- maximum full-head diagonal logit error: `9.53674316406e-07`
