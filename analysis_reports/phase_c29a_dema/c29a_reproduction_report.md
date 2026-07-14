# C29-A Reproduction Report

- gate: `C29A_ANALYSIS_AUTHORIZED` (`35/35`)
- scope: fixed train-fit diagnostics with validation-only route decisions
- official tolerances: max absolute logit error <= 1e-6 and probability error <= 1e-7
- temporal weights are checked both against the exact official formula and saved latest-slot values
- all C27 checkpoint tensors must remain bitwise unchanged

| seed | train n | val n | max logit error | max prob error | temporal formula error | decomposition error | unchanged | pass |
|---:|---:|---:|---:|---:|---:|---:|---|---|
| 0 | 602 | 94 | 8.881784197e-16 | 1.11022302463e-16 | 0 | 0 | True | True |
| 42 | 602 | 94 | 8.881784197e-16 | 1.11022302463e-16 | 0 | 0 | True | True |
| 3407 | 602 | 94 | 4.4408920985e-16 | 1.11022302463e-16 | 0 | 0 | True | True |
