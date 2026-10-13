# PMD implementation

The package name `isdynamic` is retained for compatibility with the tested code.

| Directory | Responsibility |
|---|---|
| `core/` | Token datasets, pair sampling, probe models, training, and feature transformations |
| `models/` | Frozen-backbone loading and intermediate-feature adapters |
| `export/` | Shared token-export entrypoints |
| `probes/` | Shared training/sweep entrypoints |
| `viz/` | Dense mask inference, refinement, evaluation, and existing figures |
| `release/` | Portable table/figure generation and reference imports |
| `baselines/` | Optional PCA+k-means no-probe baseline |
| `legacy/` | Validated per-backbone implementations called by shared entrypoints |

Public configurations are under `configs/probes/`. Public commands are under `scripts/`; see the root README for the reproduction workflow. The legacy implementations preserve the tested model-specific feature extraction.
