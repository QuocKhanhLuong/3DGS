# SDF-Manifold Active Gaussian Memory

Research repository for an ISBI-oriented medical imaging project that builds a patient-specific Gaussian representation from sparse, adaptively selected multi-sequence MRI observations.

## Core idea

The method links two main contributions:

1. **SDF/3D-SLNR-adaptive low-DoF Gaussian prior** — an anatomical SDF scaffold derives Gaussian orientation and constrains position/covariance instead of optimizing free position–quaternion–scale parameters.
2. **Balanced multi-wave observability routing** — parallel information fronts select complementary sequence–slice observations over an evolving feature–geometry manifold and stop at an observability fixed point.

```text
SDF scaffold
    → low-DoF Gaussian memory
    → observability graph
    → balanced multi-wave routing
    → sparse sequence–slice queries
    → local evidence assimilation
    → segmentation / reconstruction / uncertainty
```

## Start here

- [`MASTER_KNOWLEDGE.md`](MASTER_KNOWLEDGE.md) — consolidated research conclusions and mathematical formulation.
- [`architecture.md`](architecture.md) — block-level architecture, state definitions, interfaces, and MVP/full variants.
- [`pipeline.md`](pipeline.md) — end-to-end training and inference flow, pseudocode, losses, evaluation, and implementation order.
- [`knowledge/`](knowledge/) — detailed knowledge package split by topic.
- [`knowledge/figures/pipeline_v1.webp`](knowledge/figures/pipeline_v1.webp) — current conceptual pipeline figure.

## Knowledge package

| File | Scope |
|---|---|
| `01_problem_and_task.md` | Problem definition, task setting, input/output |
| `02_end_to_end_pipeline.md` | High-level system pipeline |
| `03_novelty_1_sdf_gaussian.md` | SDF-constrained low-DoF Gaussian prior |
| `04_novelty_2_multiwave_trajectory.md` | Balanced multi-wave trajectory optimization |
| `05_convergence_and_stopping.md` | Fixed-point convergence and stopping criteria |
| `06_training_and_losses.md` | Training stages and objectives |
| `07_evaluation_protocol.md` | Metrics, baselines, and ablations |
| `08_implementation_blueprint.md` | Software implementation blueprint |
| `09_risks_and_reviewer_questions.md` | Technical risks and reviewer-facing concerns |
| `10_isbi_paper_outline.md` | Draft ISBI paper structure |
| `11_reference_3d_slnr_analysis.md` | 3D-SLNR analysis and transferable insights |
| `pseudocode/algorithm.md` | Algorithm-level pseudocode |

## Current recommended task

**Budgeted active multi-sequence 3D segmentation from sparse slice queries**, with held-out slice reconstruction as auxiliary supervision.

The full-volume data are available only as hidden training/evaluation supervision. During an episode, the model may observe only the sequence–slice queries selected by the trajectory controller.

## Implementation priority

1. registered MRI slice provider and physical metadata;
2. plane-aware Gaussian renderer;
3. SDF-constrained low-DoF Gaussian state;
4. fixed-anchor initialization and local assimilation;
5. segmentation and auxiliary reconstruction;
6. uncertainty/observability state;
7. single-wave routing baseline;
8. balanced multi-wave scheduler;
9. convergence controller;
10. adaptive topology and local graph repair.

## Status

Research design and implementation planning stage. Claims such as near-full-volume performance from a small query budget remain hypotheses to be validated experimentally.
