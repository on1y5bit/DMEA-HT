# Evidence BCE Failure Report

## Why C1 Initially Looked Promising

C1 text morphology BCE initially reported validation AUC 0.7782 +/- 0.0350, above the strict MVP validation reference 0.7581 +/- 0.0171. This made it a reasonable candidate for follow-up, but only under a validation/stability gate.

## Why C1 Cannot Remain The Main Candidate

Phase C4 expanded the seed set and found mean validation AUC 0.7718, std 0.0278, median 0.7868, and min/max 0.7379 / 0.8040. Multiple bad seeds fell below the strict MVP reference, so the branch failed stability checking.

## Why C2 Does Not Rescue It

C2 text-anchor variants did not replace C1 or strict MVP as a stable validation path. They remain failed ablations and should not be used to re-promote weak evidence supervision.

## Why C6 Does Not Rescue It

C6 tested lower and delayed text morphology BCE on bad seeds only. The best candidate, delay_w001_start5, reached validation AUC 0.7450 +/- 0.0070, below the strict MVP reference. No C6 candidate reached STABILIZATION_PASS_RECOMMEND_FORMAL.

## Shortcut Residual Interpretation

C5 found bad-seed max shortcut residual Spearman around 0.2291, while the prior good-seed reference was around 0.2744. This does not support residual selected-structure shortcut coupling as the primary failure cause.

## Likely Failure Mode

C5 localized the failure to optimization/checkpoint instability and positive-probability suppression: good seed mean validation AUC 0.7933 versus bad seed mean validation AUC 0.7430. Bad seeds helped negatives relative to MVP but harmed positive-label patients.

## Future Evidence Use

Evidence labels should be used as analysis variables, explanation metadata, stratification fields, and patient-level evidence reporting aids. They should not be used as a direct BCE training target unless a new positive-preserving formulation is explicitly justified and pilot-gated.
