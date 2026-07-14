# C29-A Frozen Representation Inventory

- S0-S5 were taken from the real frozen C27 forward path under inference mode.
- Exact pre-projection order: five 256-dimensional mechanism states, five conflict scalars, then the 256-dimensional frozen fallback bio context.
- Patient projection order: `Linear(1541, 256) -> GELU -> LayerNorm(256)`.
- Classifier formula: `weight dot h_patient + bias`; the preceding Dropout is inactive in evaluation mode.
- Only patient-level float representations and scalar diagnostics were retained.

| stage | validation shape | description |
|---|---|---|
| S0_temporal_mechanisms | `[94, 5, 256]` | five official temporal mechanism states |
| S1_conflicts | `[94, 5]` | five fixed latest-history conflict scalars |
| S2_pre_projection | `[94, 1541]` | S0 flattened, then S1, then frozen fallback bio context |
| S3_projection_linear | `[94, 256]` | patient projection Linear output before GELU |
| S3_projection_post_gelu | `[94, 256]` | patient projection post-GELU before LayerNorm |
| S4_patient_state | `[94, 256]` | official LayerNorm output and classifier input |
| S5_classifier_dot | `[94, 1]` | classifier weight dot official patient state |
