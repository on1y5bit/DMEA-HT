# C31-A Visit-Text Role Graph

The graph below is reconstructed from the frozen C27/C30 source path.
Every active group switches only its mapped text-derived mechanism source.

- `R1_MORPHOLOGY_SUPPORT_GROUP`: nodes `support[0]+nonspecific[3] mean` -> `M1`; available `True`; Consumed by C27 morphology source.
- `R4_OPPOSITION_GROUP`: nodes `opposition[1]` -> `M4`; available `True`; Consumed by C27 opposition source.
- `R5_TEMPORAL_GROUP`: nodes `uncertainty[2]+temporal[4] mean` -> `M5`; available `True`; Consumed by C27 temporal text source.
- `GLOBAL_NODE_UNAVAILABLE`: nodes `global[5]` -> `none`; available `False`; Projected but not consumed by a C27 mechanism slot.

Guided pooling falls back to the full attention mask when its role mask is absent.
The temporal node uses latest/history projection when either temporal mask is present.
