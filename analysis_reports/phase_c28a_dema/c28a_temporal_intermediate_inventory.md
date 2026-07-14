# C28-A Temporal Intermediate Inventory

- Intermediates were captured with a temporary forward pre-hook on the frozen C27 core.
- The hook captured the exact source states, source-valid mask, visit mask, and fallback bio context used by the official forward.
- Visit mechanism states, content scores, ordinal recency, combined scores, temporal weights, and conflicts were obtained under inference mode.
- Counterfactuals reused the same visit-state, conflict, fallback-context, patient-projection, and classifier tensors; only temporal weights changed.
- Raw images, token tensors, and full visit-state tensors were not exported.

| seed | patients | max visits | slots | hidden | content range | combined range |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 94 | 6 | 5 | 256 | [-1.841630, 3.153718] | [-1.718890, 3.245839] |
| 42 | 94 | 6 | 5 | 256 | [-1.204976, 2.020634] | [-1.117316, 2.693393] |
| 3407 | 94 | 6 | 5 | 256 | [-2.579194, 1.218430] | [-2.405907, 1.787172] |
