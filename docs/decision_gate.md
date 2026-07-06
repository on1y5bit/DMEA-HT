# DMEA-HT Decision Gate

This gate is used before promoting any future DMEA-HT candidate beyond the current main result.

## Non-negotiable Rules

1. Patient-level split must remain unchanged.
2. Labels, task definition, image paths, report text, bio values, and manifest split assignment must remain unchanged unless a new data audit phase explicitly approves the change.
3. Test metrics are reporting-only and must never decide model selection.
4. Main promotion metric is validation AUC.
5. Shortcut variables must not be fed into the classifier.

## Required Evidence

Every candidate must provide:

- static compile result for changed scripts;
- exact run directory;
- validation AUC and standard deviation;
- test AUC marked reporting-only;
- prediction CSV paths;
- shortcut residual audit over selected structural fields;
- decision-gate summary CSV.

## Promotion Criteria

A candidate can be promoted only if:

- it completes the planned run scope;
- it beats the current main candidate by validation AUC;
- shortcut residual audit does not show a new structural shortcut concern;
- any test improvement is treated as supporting context only.

## Future Training Recommendation

Do not launch a new formal training run directly from an idea. First run a small pilot or analysis gate that checks whether the candidate has a plausible validation-AUC benefit and does not increase shortcut reliance. Promote to formal training only after that gate passes.
