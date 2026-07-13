# Corrected Next Step: Phase C16-MEA Design Audit

## Status

The mistaken C16 DSSA route is stopped and reverted. The only permitted next phase is:

```text
Phase C16-MEA
Disease-Mechanism and Evidence-Aware Alignment
```

No new model implementation or training may start until the design audit below is complete and reviewed.

## Required Audit Deliverables

Create under `analysis_reports/phase_c16_mea_design/`:

- `c16_mea_existing_model_evidence_mapping.md`
- `c16_mea_available_input_fields.csv`
- `c16_mea_available_bio_semantics.md`
- `c16_mea_evidence_role_definition.md`
- `c16_mea_mechanism_graph_definition.md`
- `c16_mea_proposed_modules.md`
- `c16_mea_loss_design.md`
- `c16_mea_shortcut_exclusion_checklist.md`
- `c16_mea_design_review.md`

## Audit Questions

1. Which evidence roles already exist in C13 and C14?
2. Which roles can be derived without weak-label leakage?
3. Which biochemical values and semantics are truly available?
4. Which HT mechanism relations are medically defensible from those fields?
5. Which relations can be trained with patient labels only?
6. Which proposed roles require dictionaries or weak labels and must therefore remain blocked?
7. How are support, opposition, uncertainty, and conflict kept distinct?
8. How is temporal evidence represented while preserving C13 report construction?
9. How are all shortcut and missingness-count fields excluded from model inputs and losses?
10. What exact evidence-mechanism failure in C13 is each proposed module intended to repair, and why should that improve validation AUC?

## Allowed Design Shape

```text
modality encoder
  -> modality evidence tokens
  -> evidence-role assignment
  -> HT mechanism relation module
  -> evidence-state aggregation
  -> disease-state alignment
  -> patient-level classifier
```

Candidate names may include `ImageEvidenceProjector`, `TextEvidenceRoleProjector`, `BioEvidenceProjector`, `HTMechanismRelationLayer`, `EvidenceConflictAggregator`, and `DiseaseStateAlignmentHead`. These names do not authorize implementation before the audit is approved.

## Training Gate After Audit

If and only if the design audit passes:

1. Run a seed-0 smoke.
2. Run a full seed-0 pilot only after the smoke passes.
3. Run seeds `42` and `3407` only after seed 0 passes the C13 comparison and safety gates.

C13 remains the fallback and strict-best baseline. Test data cannot select the design, checkpoint, threshold, loss, or route.
