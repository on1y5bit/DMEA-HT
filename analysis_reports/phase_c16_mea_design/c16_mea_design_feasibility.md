# C16-MEA Design Feasibility

Design status: `C16_MEA_DESIGN_AUDIT_PASS_WITH_CONSTRAINTS`.

This report authorizes implementation only when every hard input check passes. It does not authorize training before static/synthetic validation and both seed-0 smoke gates.

## Hard Checks

| check | pass |
| --- | --- |
| required_manifest_fields | 1.0000 |
| bio_vector_order_and_length | 1.0000 |
| real_bio_source_fields_present | 1.0000 |
| temporal_markers_available | 1.0000 |
| text_dictionaries_available | 1.0000 |
| c14_diagnostics_available | 1.0000 |
| shortcut_map_complete | 1.0000 |
| no_mistaken_dssa_symbols | 1.0000 |

## Bio Semantics

Use verified `TgAb/TPOAb` as an immune-observed group, `FT3/FT4/TSH` as a thyroid-function-observed group, and `sex/age` as other observed context. Values remain continuous observed evidence with per-field validity masks. Immune-field coverage is sparse (TgAb=0.0564, TPOAb=0.0538). An immune node may participate only for observed values; missing fields or whether an antibody test was ordered must not become support, opposition, reliability, or gate evidence.

Reference-range columns available: `False`. Rows with trusted abnormal metadata: `0`. Therefore no abnormal, normal, support, or opposition rule may be derived from bio values.

| bio_index | field_name | semantic_group | manifest_observed_fraction | source_column_present | reference_range_available | abnormal_flag_one_count |
| --- | --- | --- | --- | --- | --- | --- |
| 0.0000 | sex | other_observed | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 1.0000 | age | other_observed | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 2.0000 | TgAb | immune_observed | 0.0564 | 1.0000 | 0.0000 | 0.0000 |
| 3.0000 | FT3 | thyroid_function_observed | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 4.0000 | FT4 | thyroid_function_observed | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 5.0000 | TPOAb | immune_observed | 0.0538 | 1.0000 | 0.0000 | 0.0000 |
| 6.0000 | TSH | thyroid_function_observed | 1.0000 | 1.0000 | 0.0000 | 0.0000 |

| semantic_group | field_names | bio_indices | all_source_fields_present | mean_manifest_observed_fraction | any_reference_range_available | trusted_abnormal_metadata_rows | implementation_path | blocked_claim |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| other_observed | sex/age | 0/1 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | group observed continuous values; learn evidence roles from patient supervision | no abnormal/normal/support/opposition claim without verified ranges |
| immune_observed | TgAb/TPOAb | 2/5 | 1.0000 | 0.0551 | 0.0000 | 0.0000 | group observed continuous values; learn evidence roles from patient supervision | no abnormal/normal/support/opposition claim without verified ranges |
| thyroid_function_observed | FT3/FT4/TSH | 3/4/6 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | group observed continuous values; learn evidence roles from patient supervision | no abnormal/normal/support/opposition claim without verified ranges |

## Temporal Evidence

Build character-position masks from explicit C13 latest/history/full-report markers, and use learned fallback pooling when a section or role mask is absent.

Latest marker coverage: `0.9821`; history marker coverage: `0.5410`; full-report marker coverage: `0.9885`.

| item_type | field_or_construct | rows_present_or_derivable | fraction_present_or_derivable | rows_visible_in_model_window | semantics | allowed_c16_mea_use | restriction |
| --- | --- | --- | --- | --- | --- | --- | --- |
| derived_temporal_state | latest_support_or_opposition | 766.0000 | 0.9821 | NA | dictionary masks inside the explicit latest-focus section | masked evidence pooling with learned fallback when the section is absent | absence is unavailable evidence, not negative disease evidence |
| derived_temporal_state | historical_support_or_opposition | 422.0000 | 0.5410 | NA | dictionary masks inside the explicit history-focus section | masked evidence pooling with learned fallback when the section is absent | absence is unavailable evidence, not negative disease evidence |
| derived_temporal_state | latest_history_conflict | 422.0000 | 0.5410 | NA | opposing dictionary evidence across explicit latest/history sections | masked evidence pooling with learned fallback when the section is absent | absence is unavailable evidence, not negative disease evidence |
| derived_temporal_state | full_report_fallback | 771.0000 | 0.9885 | NA | learned pooling over the unchanged full-report section | masked evidence pooling with learned fallback when the section is absent | absence is unavailable evidence, not negative disease evidence |

## C14 Evidence Basis

| phase | path | available | row_count | purpose | restriction | route_or_status |
| --- | --- | --- | --- | --- | --- | --- |
| C14-A | analysis_reports/phase_c14a/c14a_positive_patient_token_exposure_val.csv | 1.0000 | 141.0000 | text evidence exposure and latest/full/first-window term counts | audit stratification only; never a model target or selector | NA |
| C14-B | analysis_reports/phase_c14b/c14b_representation_diagnostics_val.csv | 1.0000 | 282.0000 | C13 representation, contribution, evidence-score, and discordance diagnostics | validation diagnostics only; do not feed saved diagnostics into C16-MEA | NA |
| C14-B | analysis_reports/phase_c14b/c14b_modality_masking_val.csv | 1.0000 | 282.0000 | diagnostic modality masking and single-modality-like probabilities | distribution-shift diagnostic, not a training ablation target | NA |
| C14-B | analysis_reports/phase_c14b/c14b_text_occlusion_val.csv | 1.0000 | 141.0000 | diffuse, negative, and temporal-prefix text occlusion diagnostics | validation diagnostic only; no patient labels derived from occlusion | NA |
| C14-C | analysis_reports/phase_c14c/c14c_pairwise_inversions_by_seed.csv | 1.0000 | 6627.0000 | validation positive-negative inversion inventory | reporting/audit only; no validation pairs in training | NA |
| C14-D | analysis_reports/phase_c14d/c14d_hard_patient_profiles.csv | 1.0000 | 79.0000 | hard-patient multimodal and evidence profile | must not become sample weights, model inputs, or route labels | NA |
| C14-E | analysis_reports/phase_c14e/c14e_candidate_mechanism_coverage.csv | 1.0000 | 3.0000 | candidate mechanism coverage and matched-control generalizability | failed generalizability gate; cannot supervise or justify a broad fix | NA |
| C14-E | analysis_reports/phase_c14e/c14e_route_gate_summary.csv | 1.0000 | 1.0000 | final C14 evidence gate | C16-MEA is a separately authorized hypothesis and must retain this limitation | route=DATA_LIMIT_NO_GENERAL_MODEL_FIX;allowed_next_step=KEEP_C13_AND_REPORT_LIMITATION; |

C14-A showed that relevant text evidence is generally exposed. C14-B found no single stable global modality-removal or fusion mechanism. C14-C/D localized many failures to hard patient subgroups, and C14-E failed the generalizability gate. C16-MEA therefore targets evidence organization and conflict handling, but it must not claim that C14 proves a general mechanism.

## Proposed Modules

| module | verified_inputs | permitted_output | expected_auc_mechanism | constraint |
| --- | --- | --- | --- | --- |
| ImageMorphologyEvidenceProjector | per-image tokens + image_mask | learnable morphology-role slots and global fallback | retain multiple morphology patterns instead of a single image mean | patient supervision only; slot names are architectural, not finding labels |
| TextEvidenceRoleProjector | character tokens + report mask + audited dictionary/temporal position masks | support, opposition, uncertainty, nonspecific, temporal, global evidence | separate coexisting support and opposition that C14 found visible but inconsistently used | masks guide pooling only; learned fallback for every empty mask; no weak-label BCE |
| BioEvidenceProjector | seven ordered continuous values + per-field validity mask | immune-observed, thyroid-function-observed, and other-observed evidence | preserve verified biochemical group structure without missingness counts | no abnormal/normal direction without reference ranges; role direction remains latent |
| HTMechanismRelationLayer | audited modality evidence nodes | M1 morphology, M2 immune-observed, M3 function-observed, M4 opposition, M5 temporal, M6 disease state | relate evidence through named HT mechanisms rather than raw-modality similarity | first alignment loss is image-text morphology only; no bio-text alignment without matching text semantics |
| EvidenceConflictAggregator | support/opposition/uncertainty nodes + reliability masks | separate support, opposition, uncertainty, and conflict states | avoid averaging contradictory evidence highlighted by C14 | high conflict downweights alignment; modality availability is not a disease scalar |
| DiseaseStateAlignmentHead | mechanism state + evidence states + conflict state | binary patient HT logit and diagnostics | order patient support versus opposition while preserving ambiguous cases | binary task only; internal states are not new labels |

## Loss Contract

| loss | weight | status | evidence_contract |
| --- | --- | --- | --- |
| L_cls | 1.0000 | required | patient-level binary label |
| L_state_margin | 0.0300 | allowed | patient support-opposition ordering from training label |
| L_mechanism_alignment | 0.0200 | allowed_with_scope | image-text morphology only, valid-pair and low-conflict weighted |
| L_role_separation | 0.0050 | allowed | clinical support versus opposition states |
| L_rank | 0.0200 | variant_B_only | training-batch positive-negative pairs only |

Auxiliary weights use three classification-only warmup epochs and ramp through epoch 7. No broad sweep is allowed. Core and Core+Ranking are the only variants.

## Shortcut Exclusion

| field | item_type | allowed_as_model_input | allowed_as_loss_or_gate_input | implementation_rule |
| --- | --- | --- | --- | --- |
| n_images | shortcut_or_structural_metadata | 0.0000 | 0.0000 | retain only in batch shortcuts/export tables; never tensorize for C16-MEA |
| n_visits | shortcut_or_structural_metadata | 0.0000 | 0.0000 | retain only in batch shortcuts/export tables; never tensorize for C16-MEA |
| selected_n_visits | shortcut_or_structural_metadata | 0.0000 | 0.0000 | retain only in batch shortcuts/export tables; never tensorize for C16-MEA |
| raw_n_visits | shortcut_or_structural_metadata | 0.0000 | 0.0000 | retain only in batch shortcuts/export tables; never tensorize for C16-MEA |
| used_images | shortcut_or_structural_metadata | 0.0000 | 0.0000 | retain only in batch shortcuts/export tables; never tensorize for C16-MEA |
| raw_n_images | shortcut_or_structural_metadata | 0.0000 | 0.0000 | retain only in batch shortcuts/export tables; never tensorize for C16-MEA |
| has_bio | shortcut_or_structural_metadata | 0.0000 | 0.0000 | retain only in batch shortcuts/export tables; never tensorize for C16-MEA |
| bio_missing_count | shortcut_or_structural_metadata | 0.0000 | 0.0000 | retain only in batch shortcuts/export tables; never tensorize for C16-MEA |
| report_length | shortcut_or_structural_metadata | 0.0000 | 0.0000 | retain only in batch shortcuts/export tables; never tensorize for C16-MEA |
| source_folder | shortcut_or_structural_metadata | 0.0000 | 0.0000 | retain only in batch shortcuts/export tables; never tensorize for C16-MEA |
| image_padding_count | shortcut_or_structural_metadata | 0.0000 | 0.0000 | retain only in batch shortcuts/export tables; never tensorize for C16-MEA |
| padding_count | shortcut_or_structural_metadata | 0.0000 | 0.0000 | retain only in batch shortcuts/export tables; never tensorize for C16-MEA |
| selected_visit_dates | shortcut_or_structural_metadata | 0.0000 | 0.0000 | retain only in batch shortcuts/export tables; never tensorize for C16-MEA |
| phase_c13_focus_prefix_chars | shortcut_or_structural_metadata | 0.0000 | 0.0000 | retain only in batch shortcuts/export tables; never tensorize for C16-MEA |
| phase_c13_n_latest_focus_clauses | shortcut_or_structural_metadata | 0.0000 | 0.0000 | retain only in batch shortcuts/export tables; never tensorize for C16-MEA |
| phase_c13_n_history_focus_clauses | shortcut_or_structural_metadata | 0.0000 | 0.0000 | retain only in batch shortcuts/export tables; never tensorize for C16-MEA |
| phase_c13_n_visits_parsed | shortcut_or_structural_metadata | 0.0000 | 0.0000 | retain only in batch shortcuts/export tables; never tensorize for C16-MEA |
| image_mask | validity_mask | 1.0000 | 1.0000 | mask padded image tokens before evidence pooling |
| report_attention_mask | validity_mask | 1.0000 | 1.0000 | mask padded text tokens and empty dictionary masks |
| bio_missing_mask | validity_mask | 1.0000 | 1.0000 | mask unavailable bio values per field; do not sum or encode the count |

## Explicitly Blocked

- No shared/private, modality-invariant, DecAlign, or generic orthogonality modules.
- No report-derived image labels or old morphology BCE losses.
- No rule-based bio abnormality, reference range, or support/opposition target.
- No immune/function cross-modal alignment unless matching text semantics are separately verified.
- No shortcut counts, source folder, report length, modality-presence scalar, or C14 hard-group field in the predictor.
- No test-based architecture, variant, checkpoint, threshold, loss, or route selection.

## Exact Expected AUC Mechanism

The proposed change is intended to improve patient ranking by keeping visible supporting and opposing evidence separate, representing temporal contradiction explicitly, and relating only clinically compatible evidence through named mechanism nodes. This addresses the observed failure of evidence use and pairwise ordering without assuming that one modality should be globally removed or that raw modalities should share a representation.

## Next Gate

If this design status passes, implementation may begin with backward-compatible modules and static/synthetic checks. Training remains blocked until both Core and Core+Ranking seed-0 smoke configurations pass all collapse, saturation, compatibility, and shortcut checks.
