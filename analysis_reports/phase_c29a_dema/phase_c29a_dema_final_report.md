# DEMA-HT Phase C29-A Final Report

## Execution

- canonical server directory: `/home/linruixin/chen/project/DMEA-HT`
- branch: `main`
- starting commit: `60d090b`
- analysis commit: `c3a8e5f`
- runtime gate: `C29A_ANALYSIS_AUTHORIZED` (`35/35`)
- analysis-only; fixed train-fit statistical probes; validation-only route decision
- no neural parameter update, checkpoint write, calibration, prediction averaging, or seed selection

## Probe Summary

| probe | val AUC mean | std | gain vs official | improved seeds | sensitivity | inversions | material damage | max gap | random max | safety |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| P0_official_C27 | 0.8822996831 | 0.0168958421 | +0.0000000000 | 0/3 | 0.794326 | 260.000 | 42 | nan | nan | True |
| P1_patient_state | 0.8242040139 | 0.0355251467 | -0.0580956692 | 0/3 | 0.695035 | 388.333 | 52 | 0.175975 | 0.506564 | False |
| P2_pre_projection | 0.8103214124 | 0.0150209688 | -0.0719782707 | 0/3 | 0.737589 | 419.000 | 48 | 0.179584 | 0.541874 | False |
| P3_temporal_mechanisms | 0.7953825260 | 0.0115503403 | -0.0869171571 | 0/3 | 0.744681 | 452.000 | 49 | 0.196509 | 0.538253 | False |
| P4_conflicts_negative_control | 0.6923192998 | 0.0172340927 | -0.1899803833 | 0/3 | 0.531915 | 551.667 | 91 | -0.103620 | 0.461747 | False |
| P5_C17_mechanism_reference | 0.7769729893 | 0.0322981313 | -0.1053266938 | 0/3 | 0.680851 | 492.667 | 66 | 0.175289 | 0.516523 | False |

## Head Geometry And Safety

- classifier weight norms by seed: `0.580942, 0.592173, 0.575182`
- classifier biases by seed: `-0.058886, -0.009183, 0.040776`
- classifier-centroid direction cosine by seed: `0.557929, 0.622011, 0.478339`
- shortcut-safe formal objects: `33/36`; authorization is evaluated per candidate probe.
- raw visit/image associations remain separate audit warnings and are not probe inputs.

## Decision

- `C29A_VISIT_REPRESENTATION_LIMITATION_SUPPORTED`
- `C29B_NOT_AUTHORIZED`
- current strict best: `DEMA_C17_POSITIVE_PRESERVATION`
- diagnostic probes and swaps are not formal models.
