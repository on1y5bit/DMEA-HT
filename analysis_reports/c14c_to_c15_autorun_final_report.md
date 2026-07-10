# DMEA-HT C14-C To C15 Autonomous Run Final Report

## C14-C Status

- Server commit: `238227a`.
- C13 manifest and formal seeds `[0, 42, 3407]` were used.
- Reproduction passed for all seeds: 94 / 94 validation rows, matching IDs and labels.
- Maximum absolute probability difference: `1.11e-16` for every seed.
- Pairwise coverage: `6627 = 2209 x 3` positive-negative pairs.
- Total inversion rows: `885`.
- All-seed inversion pairs: `215`.
- Majority-seed inversion pairs: `75`.
- Single-seed inversion pairs: `90`.

## Route Decision

`HARD_PATIENT_SUBGROUP_FAILURE`

Final status: `C14C_HARD_SUBGROUP_STOP`.

The top-five patient inversion share was `59.32%`, with several patients producing inversions across all three seeds. Image-opposed inversions were `28.55%`, below the 30% image route threshold, and image masking repaired the margin in zero seeds. Fusion-interaction inversions were `8.55%`. These results do not justify an image-correction or fusion-residual training route.

## C15 Decision

C15 was not authorized and no training was launched. The allowed next step is `MORE_ANALYSIS_ONLY`: perform a focused hard-patient/subgroup audit before considering any new model change.

C13 remains the current strict best at validation AUC `0.8665 +/- 0.0077`. No test result was used for route selection, and no AUC 0.90 claim is made.

Detailed C14-C artifacts are in `analysis_reports/phase_c14c/`.

## C14-D Follow-Up

C14-D confirmed the hard-patient concentration without authorizing training:

- 79 all-seed hard patients: 43 negative and 36 positive;
- top-20 patient-side inversion incidence share: 66.27%;
- negative hard patients were more image-opposed, while positive hard patients were more text-driven;
- no single correction mechanism met a new training gate.

C14-D route: `HARD_PATIENT_SUBGROUP_AUDIT_CONFIRMED`.

C15 remains unauthorized. The next valid action is a manual/clinical audit of the highest-impact positive and negative patients.
