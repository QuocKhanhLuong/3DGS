# T0 adversarial review

Date: 2026-07-29  
Scope: Legal Physical Forward Operator  
Final verdict: **PASS**

## Review history

The first independent review blocked T0 on a public raw-provider bypass,
density-weighted slab semantics, renderer numeric closure, and inability to
represent valid left-handed source geometry. It also warned about budget and
capability replay, scale-dependent affine checks, mutable tensors, and
unsupported-region semantics.

The implementation was revised to:

- keep raw payload access and content digests private to a manifest-bound
  provider;
- require commit plus a ledger-bound, single-use capability for target reveal;
- use exact decimal target-budget accounting;
- evaluate normalized latent intensity independently at every slab depth;
- report weighted supported PSF mass and enforce a named coverage threshold;
- revalidate Gaussian tensors on every render and enforce dtype-aware numeric
  bounds;
- validate source-affine rank after overflow-safe column normalization;
- preserve independently signed slice axes for left-handed source affines.

A second review found and blocked a pre-commit side channel: target content
digests were still present in public observation metadata and affected the
canonical manifest hash. Digests were removed from both surfaces and retained
only as private provider integrity data.

## Final independent findings

- Public metadata and canonical manifest serialization contain no content
  digest; different private digests produce the same public manifest hash.
- Weighted PSF coverage, supported-sample renormalization, and the configured
  support threshold are implemented and regression-tested.
- Extreme finite affine scales pass the overflow-safe rank check, while
  singular and ill-conditioned blocks fail.
- `0.1 + 0.2` target costs fit an exact `0.3` budget.
- Full regression gate: 44 tests passed, with 26 additional subtests.

No remaining blocker was identified for T0.

## Residual boundary

The observation ledger is an in-process scientific-validity contract, not a
security sandbox. Fully sampled audit volumes and hostile-code isolation still
require a separate process that receives only serialized reconstructions.
