# Phase C7 Final Report

## Objective

Phase C7 is a decision-consolidation and route-correction phase. It does not train a new model and does not promote a new candidate.

## Inputs Used

| path | status | notes |
| --- | --- | --- |
| /home/linruixin/chen/project/DMEA-HT/analysis_reports/phase_c3/model_comparison_table.csv | loaded | 7 rows |
| /home/linruixin/chen/project/DMEA-HT/analysis_reports/phase_c3/decision_gate_summary.csv | loaded | 7 rows |
| /home/linruixin/chen/project/DMEA-HT/analysis_reports/phase_c4/c1_extended_seed_summary.csv | loaded | 7 rows |
| /home/linruixin/chen/project/DMEA-HT/analysis_reports/phase_c4/c1_weight_pilot_summary.csv | loaded | 6 rows |
| /home/linruixin/chen/project/DMEA-HT/analysis_reports/phase_c5/c1_seed_failure_summary.csv | loaded | 7 rows |
| /home/linruixin/chen/project/DMEA-HT/analysis_reports/phase_c5/c1_vs_mvp_patient_delta_val.csv | loaded | 658 rows |
| /home/linruixin/chen/project/DMEA-HT/analysis_reports/phase_c5/c1_seed_shortcut_residual.csv | loaded | 42 rows |
| /home/linruixin/chen/project/DMEA-HT/analysis_reports/phase_c6/c6_badseed_pilot_summary.csv | loaded | 3 rows |
| /home/linruixin/chen/project/DMEA-HT/analysis_reports/phase_c6/c6_positive_preservation.csv | loaded | 18 rows |
| /home/linruixin/chen/project/DMEA-HT/analysis_reports/phase_c6/c6_final_report.md | loaded | 1957 chars |
| /home/linruixin/chen/project/DMEA-HT/analysis_reports/phase_c6/c6_badseed_pilot_summary.csv | loaded_appendix | 3 rows |
| /home/linruixin/chen/project/DMEA-HT/analysis_reports/phase_c6/c6_decision_gate_summary.csv | loaded_appendix | 3 rows |
| /home/linruixin/chen/project/DMEA-HT/analysis_reports/phase_c6/c6_epoch_dynamics.csv | loaded_appendix | 211 rows |
| /home/linruixin/chen/project/DMEA-HT/analysis_reports/phase_c6/c6_positive_preservation.csv | loaded_appendix | 18 rows |
| /home/linruixin/chen/project/DMEA-HT/analysis_reports/phase_c6/c6_shortcut_residual_audit.csv | loaded_appendix | 54 rows |

## Main Decision

Current main path is `strict_structural_matched_dmea_mvp`. The weak text morphology BCE branch is demoted to ablation-only / unstable.

## Why C1 Was Demoted

C1 initially looked promising at validation AUC 0.7782 +/- 0.0350, but Phase C4 extended-seed validation failed with mean 0.7718, std 0.0278, and min/max 0.7379 / 0.8040. Phase C5 then showed that bad seeds compressed positive-negative separation and harmed positive-label patients relative to MVP.

## Why C2/C6 Were Not Promoted

C2 text-anchor variants did not rescue the evidence branch. C6 low-weight and delayed text morphology BCE pilots produced no PASS candidate; the best partial C6 validation AUC was 0.7450 +/- 0.0070, still below the strict MVP reference.

## Current Main Path

Strict structural matched DMEA-MVP remains the stable reference with validation AUC 0.7581 +/- 0.0171. Test AUC 0.7729 +/- 0.0363 is reporting-only.

## Updated Decision Gate

Any future weak-evidence-supervised candidate must pass the positive-preservation gate and bad-seed pilot before formal evaluation. Validation and stability failures cannot be overridden by test metrics.

## What Not To Do Next

- Do not continue optimizing text morphology BCE as the main path.
- Do not enable image morphology BCE, counterfactual loss, matched SupCon, new anchor-fusion losses, or other disabled evidence losses from this failed branch.
- Do not launch formal training from C6 candidates.
- Do not promote any model based on a single good seed.

## Recommended Next Direction

Recommended next direction: keep strict MVP as the current main path and use evidence labels only for diagnostics/explanation unless a new positive-preserving alignment formulation is separately proposed and pilot-gated.

## Decision Tables

### Main Path Decision Summary

| candidate_name | phase | status | validation_auc_mean | validation_auc_std | stability_status | positive_preservation_status | shortcut_residual_status | promotion_decision | reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Strict MVP | MVP | current_main_path / stable_reference | 0.7581 | 0.0171 | stable_reference | reference | structural matched reference | current_main_path | Strict structural matched DMEA-MVP remains the stable validation reference; test AUC is reporting-only. |
| C1 text morphology only | C1/C4/C5 | ablation_only_unstable | 0.7718 | 0.0278 | STABILITY_FAIL | fails_positive_preservation | not_primary_failure_signal | not_promoted | Extended seeds fell below strict MVP on multiple bad seeds, and C5 showed positive-label harm. |
| C1 text + image evidence | C1/C3 | failed_ablation | 0.7691 | 0.0223 | not_promoted_in_c3 | not_established | not_promoted | not_promoted | Text plus image evidence did not become the stable main candidate and image morphology BCE remains disabled for future use. |
| C2 text anchor w=0.05 | C2/C3 | failed_ablation | 0.7746 | 0.0173 | not_promoted_in_c3 | not_established | not_promoted | not_promoted | Text anchor did not rescue the evidence branch or supersede the strict MVP reference. |
| C6 delay_w001_start5 | C6 | partial_no_formal | 0.7450 | 0.0070 | STABILIZATION_PARTIAL_NEEDS_MORE_ANALYSIS | fails_positive_preservation | not_alarming | not_promoted | mean validation AUC is not close to strict MVP reference |
| C6 w001 | C6 | partial_no_formal | 0.7438 | 0.0090 | STABILIZATION_PARTIAL_NEEDS_MORE_ANALYSIS | fails_positive_preservation | not_alarming | not_promoted | mean validation AUC is not close to strict MVP reference |
| C6 w0005 | C6 | failed_no_formal | 0.7421 | 0.0135 | STABILIZATION_FAIL | less_positive_harm_but_auc_gate_failed | not_alarming | not_promoted | mean validation AUC does not beat original bad-seed C1; mean validation AUC is not close to strict MVP reference |

### Ablation Status Table

| branch | last_phase_evaluated | best_validation_auc | best_decision | allowed_future_use | forbidden_future_use | notes |
| --- | --- | --- | --- | --- | --- | --- |
| strict_mvp | C3/C7 | 0.7581 | current_main_path / stable_reference | main/reference | none | Use validation metrics for selection; test metrics remain reporting-only. |
| text_morphology_bce | C6 | 0.8040 | ablation_only_unstable | ablation/report-only | main-path promotion or formal training without a new positive-preserving formulation and gate | C4 stability failed and C5/C6 showed positive-preservation concerns. |
| text_image_morphology_bce | C3 | NA | failed_ablation | ablation/report-only | direct BCE training target in the main path | Do not enable image morphology BCE from this branch. |
| text_evidence_anchor | C2/C3 | NA | failed_ablation | ablation/report-only | promotion based on anchor variants that failed C3 | C2 did not rescue the evidence branch. |
| c6_delayed_text_morphology | C6 | 0.7450 | partial_no_formal | diagnostic-only unless new positive-preservation gate is approved | formal evaluation from current C6 candidate | Best C6 candidate by validation AUC but still below strict MVP reference. |
| c6_low_weight_text_morphology | C6 | 0.7438 | failed_or_partial_no_formal | diagnostic-only unless new positive-preservation gate is approved | formal evaluation from current low-weight candidates | Low-weight candidates did not pass the C6 stabilization gate. |

### Positive Preservation Summary

| candidate | positive_abs_error_delta_vs_mvp | negative_abs_error_delta_vs_mvp | positive_negative_gap | sensitivity | specificity | positive_preservation_decision |
| --- | --- | --- | --- | --- | --- | --- |
| C1 text morphology only bad seeds | 0.2083 | -0.1716 | 0.1430 | NA | NA | FAIL: helps negatives but substantially harms positives. |
| C1 text morphology only good seeds | 0.1427 | -0.1799 | 0.2168 | NA | NA | WARNING: better AUC seeds still show positive harm in C5. |
| C6 delay_w001_start5 | 0.1690 | -0.1442 | 0.1548 | 0.6028 | 0.7234 | FAIL: no C6 candidate passed the positive-preservation/formal gate. |
| C6 w001 | 0.1746 | -0.1525 | 0.1576 | 0.5816 | 0.7518 | FAIL: no C6 candidate passed the positive-preservation/formal gate. |
| C6 w0005 | 0.1360 | -0.1094 | 0.1531 | 0.6738 | 0.6525 | FAIL: no C6 candidate passed the positive-preservation/formal gate. |
