# C16-MEA Current C13 Model Path

## Frozen Input Contract

- Manifest: `/data/csb/DMEA-HT/HT_2025.12_25/manifest_distmatch_structmatch_evidence_v2_c13_temporal_focus.jsonl`
- Patients/rows: `780` / `780`
- Model-visible report characters: `254`
- Split/label counts: `{"test": {"0": 42, "1": 42}, "train": {"0": 301, "1": 301}, "val": {"0": 47, "1": 47}}`
- Checkpoint selection remains validation AUC only; test remains reporting-only.

## Current Path

```text
image files -> ImageEncoder -> per-image tokens + image global
C13 report text -> stable character IDs -> text tokens + text global
seven bio values -> BioEncoder -> per-field tokens + bio global
all tokens -> existing generic EvidenceRoleAlignment + PatientAnchorFusion
image/text/bio globals + patient anchor + negative token -> EvidenceConservationClassifier
```

## Available Representations

- Per-image tokens and image global: `True`
- Per-character text tokens and text global: `True`
- Per-field bio tokens and bio global: `True`
- Existing patient anchor: `True`
- Existing classifier contributions: `True`
- Character positions remain aligned with report positions: `True`

## Existing Evidence Limitation

The current generic evidence roles are `morphology, immune, function, negative, uncertain, temporal`, but their role loss is zero: `True`. They are not clinically grounded C16-MEA roles and must not be relabeled as such without a new evidence path.

C16-MEA may consume encoder tokens and verified masks, but it must align evidence through documented HT mechanisms. It must not align raw modality globals or introduce shared/private branches.

Mistaken DSSA symbols present: `[]`.
