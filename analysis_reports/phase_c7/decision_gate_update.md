# Phase C7 Decision Gate Update

Any candidate derived from weak evidence supervision must pass a positive-preservation gate before formal evaluation.

Positive-preservation gate:

1. It must not reduce positive-label predicted probabilities relative to strict MVP in a systematic way.
2. It must not improve negative-label errors while substantially worsening positive-label errors.
3. It must preserve or improve positive-negative prediction gap.
4. It must not reduce validation sensitivity without a compensating validation AUC/AUPRC gain.
5. It must pass bad-seed pilot evaluation before any formal three-seed run.

Additional rules:

- No candidate that fails extended-seed stability may be promoted based on a single good seed.
- No candidate may enter formal evaluation if it is below strict MVP reference on bad-seed pilots.
- No test metric may override validation or stability failure.
