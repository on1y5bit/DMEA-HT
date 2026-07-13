# Correct Project Mainline

## Project Title

```text
疾病机制与证据感知的桥本甲状腺炎多模态对齐预测模型

Disease-Mechanism and Evidence-Aware Multimodal Alignment
for Hashimoto's Thyroiditis Prediction
```

## Scientific Question

How can heterogeneous clinical evidence from ultrasound images, medical reports, and biochemical indicators be aligned through defensible HT disease mechanisms, rather than by directly forcing raw modality representations to become similar?

## Evidence Flow

```text
Ultrasound image
  -> thyroid morphology evidence

Medical report
  -> textual support, opposition, uncertainty, and temporal evidence

Biochemical indicators
  -> immune evidence, thyroid-function evidence, and reliability

Evidence roles
  -> mechanism-aware relations
  -> patient-level HT disease state
  -> next-year binary HT prediction
```

The C13 temporal-focus construction remains the input baseline. It preserves the task definition, prediction horizon, patient-level grouping, labels, and split assignment.

## Alignment Principles

1. Align evidence through an HT mechanism relation only when the relation is clinically and data-semantically defensible.
2. Represent supporting, opposing, uncertain, and conflicting evidence separately instead of averaging them blindly.
3. Preserve temporal evidence such as persistence, recent support, latest negative evidence, and progression or contradiction across visits.
4. Estimate reliability from valid evidence representations and tensor availability masks only.
5. Never use report length, visit count, image count, padding count, biochemical missingness count, source folder, or related audit fields as predictive inputs.
6. Keep patient-level classification loss dominant; every auxiliary objective must map to a named HT evidence role or mechanism relation.
7. Select checkpoints by validation AUC only. Test remains reporting-only.

## Explicitly Rejected Mainline Concepts

The project must not use the following as its architecture or innovation narrative:

- shared-specific or common-private decomposition;
- homogeneous-heterogeneous representation learning;
- generic modality-invariant representation learning;
- modality-specific disentanglement;
- DecAlign or DecAlign-inspired objectives;
- generic shared/private orthogonality or modality-adversarial losses.

## Current Baseline

- Strict best: `C13_TEMPORAL_FOCUS_DMEA_HT`
- Formal seeds: `[0, 42, 3407]`
- Mean validation AUC: `0.8664554097`
- Mean validation AUC standard deviation: `0.0077356304`
- C13 remains frozen and reproducible.
- The mistaken C16 DSSA smoke is invalid and has no standing in model selection or scientific reporting.
