# Verification performed in the new workspace

The original `isDynamicProbe` experiment artifacts were read as references. New verification artifacts are under `outputs/verification/`.

| Check | Result |
|---|---|
| Six backbone code revisions, interpreters, and checkpoints | Paths and locked revisions match |
| YouTube-VOS subset | 160 train / 80 validation; five sampled masks per split match the archived annotations byte-for-byte |
| All public shell scripts and Python source | Syntax checks passed |
| Six backbone probing CLIs | Loaded successfully in their corresponding installed environments |
| Probe training integration | One epoch on a tiny synthetic shard produced a checkpoint, metrics, and summary in the requested directory |
| Pi3X image inference | Two `breakdance` frames produced coarse masks, full-grid scores, metadata, and visualizations; all 18 blocks used |
| SAM2 refinement and evaluator | Both mask variants evaluated on a two-frame dataset view |
| Six-backbone diagnostic | Recomputed from exact stored metrics and evaluation JSONs, including source validation and association statistics |
| Main/runtime tables | CSV, JSON, and Markdown generation passed against archived values |
| Figure 2 | Rebuilt from numeric layer-wise logs; PDF embeds TrueType fonts |
| Public CPU tests | 43 passed; covers pair labels, shuffling, probe equation, masks, evaluator, reporting, CLI paths, padding isolation, patch manifest, and project-page asset completeness |
| Project-page videos | Rhino 90, koala 100, bird of paradise 98 frames; all source frames preserved in input, score, coarse-overlay, and refined-overlay clips |
| Project-page browser checks | All nine sequence/view switches preserve seeking position with zero measured synchronization error; playback, pipeline animation, figure modal, and mobile layout verified |
| DA3 feature-extraction patch | Applied successfully to clean pinned source files, including the required recorder helper |

Full backbone exports, full training sweeps, and all benchmark inference runs have not been repeated in this workspace. The two-frame integration numbers are smoke-test results, not paper performance results.

The current SAM2 environment emits an ABI warning for its optional CUDA hole-filling extension and uses the upstream fallback. An independent installation should build that extension for its own PyTorch environment if that post-processing is required. Short, two-frame DAVIS views also have an undefined decay statistic; J-M/J-R remain available.

Exact reference files are retained under `results/reference/`. The copied JSONs include their archived source paths. No values were read from rendered plots.
