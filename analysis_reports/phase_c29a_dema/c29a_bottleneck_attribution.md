# C29-A Bottleneck Attribution

- primary label: `C29A_VISIT_REPRESENTATION_LIMITATION_SUPPORTED`
- C29-B authorization: `C29B_NOT_AUTHORIZED`
- P1 mean AUC gain: `-0.0580956692`; material-improvement seeds: `0/3`
- P2 mean AUC gain: `-0.0719782707`; P2 minus P1: `-0.0138826015`
- P3 mean AUC gain: `-0.0869171571`; material-improvement seeds: `0/3`
- official aggregate material positive damage: `42`
- P1/P2/P3 aggregate material damage: `52` / `48` / `49`
- P1/P2/P3 probability rescue counts: `14` / `20` / `20`
- classifier/full-head off-diagonal material-gain directions: `0` / `0`
- coordinate support: `{"S2_pre_projection": false, "S4_patient_state": false}`

## Prespecified Gates

- P1 final-classifier checks: `{"generalization": false, "inversion_nonworsening": false, "mean_auc_gain": false, "positive_damage_reduction": false, "random_label": true, "sensitivity_safety": false, "shortcut": true, "two_seed_direction": false}`
- P2 patient-projection checks: `{"generalization": false, "inversion_nonworsening": false, "mean_auc_gain": false, "positive_damage_reduction": false, "random_label": true, "sensitivity_safety": false, "shortcut": true, "two_seed_direction": false}`
- coordinate-mismatch rule: `False`
- visit-representation-limitation rule: `True`
- probe leakage/overfit risk: `False`
- train-validation generalization warning present: `True`
- random-label failure present: `False`
- apparent authorizing signal dependent on an unsafe gap: `False`

P4 conflict-only and P5 C17-reference probes are excluded from authorization by construction. Cross-seed swaps and classifier geometry are diagnostic and are not clinical causal evidence.
