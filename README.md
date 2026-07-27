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

- [`MASTER_KNOWLEDGE.md`](MASTER_KNOWLEDGE.md) — consolidated conclusions, formulation, convergence, task, losses, metrics, and ablations.
- [`KNOWLEDGE_PACKAGE.md`](KNOWLEDGE_PACKAGE.md) — navigable implementation and paper-development package.
- [`architecture.md`](architecture.md) — detailed block architecture, state definitions, interfaces, and MVP/full variants.
- [`pipeline.md`](pipeline.md) — end-to-end training and inference flow, pseudocode, evaluation, and implementation order.

## Current recommended task

**Budgeted active multi-sequence 3D segmentation from sparse slice queries**, with held-out slice reconstruction as auxiliary supervision.

The complete registered volumes are hidden training/evaluation supervision. During an episode, the model may access only the sequence–slice observations selected by the trajectory controller.

## Research package map

| Document | Scope |
|---|---|
| `MASTER_KNOWLEDGE.md` | Executive research synthesis and mathematical decisions |
| `KNOWLEDGE_PACKAGE.md` | Novelty packages, task, convergence, training, evaluation, ablations, and reviewer safeguards |
| `architecture.md` | Block-by-block design and code-facing interfaces |
| `pipeline.md` | Training/inference stages and implementation sequence |

## Implementation priority

1. registered MRI slice provider and physical metadata;
2. plane-aware Gaussian slice renderer;
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
