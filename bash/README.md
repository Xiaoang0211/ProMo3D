# Internal experiment backends

Use `scripts/main/run.sh` and `scripts/experiments/run.sh` as the public entrypoints. They supply portable paths, per-model interpreters, and explicit experiment settings to these retained shell backends.

This directory contains the backends needed for image-mask export, evaluation, refinement, token export, runtime measurement, source-validation evaluation, geometry consistency, and the optional PCA+k-means baseline. Duplicate model wrappers and historical migration tools were removed from the release workspace.

See `docs/main.md` and `docs/experiments.md` for commands and output locations.
