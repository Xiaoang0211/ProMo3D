# Additional experiments

All additional runs write beneath `outputs/experiments/`, grouped by experiment type. The six-backbone diagnostic uses Pi3X, Pi3, VGGT, DA3, MapAnything, and DUSt3R. Set up the two additional backbones, export their tokens, and train their main rank-8 probes first:

```bash
bash scripts/main/run.sh export vggt mapanything
bash scripts/main/run.sh train vggt mapanything
bash scripts/main/run.sh masks vggt mapanything
bash scripts/main/run.sh refine vggt mapanything
bash scripts/main/run.sh eval vggt mapanything
```

| Experiment used in the reported comparison | Command | Prerequisites |
|---|---|---|
| Correct vs within-frame shuffled labels, rank 4/8 | `bash scripts/experiments/run.sh controls pi3x da3 dust3r` | Sampled native tokens |
| Global fixed thresholds and Otsu offsets | `bash scripts/experiments/run.sh thresholds` | Pi3X continuous coarse score exports |
| Source-validation coarse J-M/J-R | `bash scripts/experiments/run.sh val_masks` | Source validation images/masks, six main rank-8 probes |
| NG/accuracy vs source/target coarse metrics | `bash scripts/experiments/run.sh diagnostic` | Six main probe logs, coarse masks/metadata, target and source-val evaluations |
| End-to-end latency and peak allocated VRAM | `bash scripts/experiments/run.sh runtime --methods pi3x geomotion flowsam seganymo` | Native environments, checkpoints, source DAVIS frames |

The shuffled-label control permutes labels within each frame group, preserves that group's label counts, and changes training labels only. Its table averages the validation metrics over every selected block and reports the always-same pair-label baseline and balanced accuracy. It does not pick a best block.

The threshold sweep applies fixed 0.3/0.4/0.5/0.6/0.7 and Otsu offsets 0.0/0.1/0.2/0.3 globally to all three target benchmarks. It reports every candidate and the requested standard deviations; it does not choose a new target-specific default.

The backbone diagnostic computes pairwise NG from stored confusion counts, using the submission's majority-pair-label normalization. It reports source-validation NG across all selected blocks, source-validation coarse J-M/J-R, target coarse metrics, Spearman/Pearson associations, and exact provenance. DAVIS token metrics are retained as post-hoc diagnostics. Source-validation metrics are used for the operational backbone comparison.

Runtime profiling uses 10 warm-up frames and 100 measurement frames (`cows` and `parkour`), includes required preprocessing and backbone inference, and reports each pipeline's own native inference configuration. Run profilers sequentially on an otherwise idle GPU. Model loading and result writes are excluded. GeoMotion, Flow-SAM, and SegAnyMo require the repositories and runtime assets listed in `configs/local.env.example`; their scripts fail if required checkpoints are absent.

Install their pinned source code with `bash scripts/setup_third_party.sh GeoMotion flowsam SegAnyMo`. The assets follow the released checkpoints: GeoMotion's `hf/best_model.pth` and RAFT weight; Flow-SAM's FlowP/FlowI decoders, SAM ViT-H/ViT-B, RAFT, and segment-anything source; SegAnyMo's `checkpoints/moseg.pth`, BootsTAPIR, and SAM2. Point `*_RUNTIME_ROOT` at these asset directories. The archived asset volume is currently unavailable on the local machine, so fresh third-party timing runs require reconnecting it or installing the assets from the upstream release instructions.

## Additional implemented ablations

These are available for further analysis; check the paper/response before attributing them to a reported table.

```bash
# Repeated-target counterfactual with the trained Pi3X probe.
bash scripts/experiments/run.sh repeated_context pi3x

# Fourier position residualization: preserves the pretrained backbone and RoPE.
bash scripts/experiments/run.sh position_residual da3 dust3r mapanything --stage train
bash scripts/experiments/run.sh position_residual da3 dust3r mapanything --stage masks
bash scripts/experiments/run.sh position_residual da3 dust3r mapanything --stage refine
bash scripts/experiments/run.sh position_residual da3 dust3r mapanything --stage eval

# Optional aspect-padding experiment; not the paper-result preprocessing.
bash scripts/experiments/run.sh padding da3 mapanything dust3r --stage export
bash scripts/experiments/run.sh padding da3 mapanything dust3r --stage train
bash scripts/experiments/run.sh padding da3 mapanything dust3r --stage masks
bash scripts/experiments/run.sh padding da3 mapanything dust3r --stage refine
bash scripts/experiments/run.sh padding da3 mapanything dust3r --stage eval
bash scripts/experiments/run.sh padding da3 mapanything dust3r --stage geometry
```

Pi3X, Pi3, and VGGT already use padding in their native PMD preprocessing; they are not included in this optional experiment. DA3, MapAnything, and DUSt3R use native preprocessing for paper reproduction and `aspect_pad` only for this additional experiment. Omitting the model list from the padding command selects these three models.

Position residualization removes feature components predictable from spatial coordinates; it is not an intervention that disables pretrained RoPE. Padding exports have their own `outputs/tokens/aspect_pad/<backbone>/` tree, and their probes, masks, and evaluations live under `outputs/experiments/padding/`. Geometry stability evaluates agreement between native and padded predictions over annotated benchmark frames; without metric-depth/pose ground truth it is a consistency measurement, not proof that absolute geometry accuracy is unchanged.

The research package also contains the PCA+k-means no-probe baseline and VGGT-Omega adapter. They remain available as extended experiments under `isdynamic/baselines/` and the internal wrappers. Neither is used to replace the six configurations in the reported backbone diagnostic.

Small historical diagnostic CSV/JSON files are preserved in `results/diagnostics/backbone_probe_diagnostic/`. Those files record the original artifact paths for provenance; reproduction commands use this workspace's paths.

Reference logs can regenerate tables and the layer-wise panel without exporting tokens or masks:

```bash
OUTPUT_ROOT="$PWD/results/reference" bash scripts/main/run.sh tables
OUTPUT_ROOT="$PWD/results/reference" bash scripts/experiments/run.sh tables --table controls
OUTPUT_ROOT="$PWD/results/reference" bash scripts/experiments/run.sh tables --table runtime
OUTPUT_ROOT="$PWD/results/reference" bash scripts/main/run.sh figures --panels layerwise
```

The reference snapshot additionally contains exact per-block logs, compact evaluation JSONs, matched-control logs, and raw timing summaries under `results/reference/`. Rebuild its tables with `isdynamic.release.tables` through the reproduction configuration. The combined reference runtime table selects the cached-SAM2 profile used in the reported comparison, and excludes earlier timing variants and smoke runs; their raw files remain available for provenance.
