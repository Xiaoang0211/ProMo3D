# Pretrained Mask Inference and Evaluation

This workflow runs a frozen backbone and trained PMD probes directly on benchmark images. It does not train a probe, export tokens, or require YouTube-VOS. The same inference, qualitative-output writer, and evaluator are used by the main reproduction pipeline.

## Run

First install the selected backbone and its checkpoint as described in [Setup](setup.md). Download the corresponding released PMD probes:

```bash
bash scripts/download_huggingface.sh --assets models --models pi3x

bash scripts/evaluate.sh pi3x \
  --probe-root outputs/pretrained/pi3x/rank_8/pair_lowrank_quadratic \
  --davis-root /path/to/DAVIS \
  --segtrack-root /path/to/SegTrackv2 \
  --fbms-root /path/to/FBMS-59 \
  --output-dir outputs/predictions/pi3x_native
```

Without explicit paths, datasets default to `datasets/` or the roots in `configs/local.env`. Without `--probe-root`, probes default to `outputs/main/probes/<backbone>/rank_8/pair_lowrank_quadratic/`. Dataset-path CLI arguments override local settings. The probe directory must preserve its group/block subfolders, each containing `config.json` and the trained checkpoint.

Add `--sam2` to generate and evaluate SAM2-refined masks as well. Install SAM2 and set its checkpoint in `configs/local.env` first. Coarse and refined results remain separate; the default evaluates only coarse masks and does not load SAM2.

Choose benchmarks explicitly when only some datasets are present:

```bash
bash scripts/evaluate.sh da3 \
  --probe-root /path/to/da3/pair_lowrank_quadratic \
  --davis-root /path/to/DAVIS \
  --benchmarks davis_all \
  --output-dir outputs/predictions/da3_native
```

The default benchmarks are `davis2017`, `segtrackv2`, and `fbms59`. DAVIS mask generation follows the existing shared DAVIS-all layout; evaluation uses the requested split. `davis2016` and `davis_all` are also supported. FBMS-59 evaluation includes only annotated frames.

For a multi-backbone bundle, `--probes-root` supports both the released `<backbone>/rank_8/pair_lowrank_quadratic/` layout and `<backbone>/pair_lowrank_quadratic/` from an existing bundle:

```bash
bash scripts/download_huggingface.sh --assets models

bash scripts/evaluate.sh pi3x pi3 vggt da3 mapanything dust3r \
  --probes-root outputs/pretrained \
  --davis-root /path/to/DAVIS \
  --segtrack-root /path/to/SegTrackv2 \
  --fbms-root /path/to/FBMS-59 \
  --output-dir outputs/predictions/all_native \
  --sam2
```

Append `--dry-run` to inspect commands without loading models or creating outputs.

## Outputs

All files stay inside the chosen output directory:

```text
run_<backbone>.json                 Dataset/probe paths and fixed inference settings
masks/<backbone>/<group>/.../       Per-sequence masks, scores, and visualizations
eval/<backbone>/<group>/<variant>/  Benchmark and per-sequence metrics in JSON
tables/motion_masks.csv            Combined J-Mean and J-Recall (%)
tables/motion_masks.json
tables/motion_masks.md
```

Each coarse frame directory contains `input_image.png`, `avg_score_upsampled_heatmap.png`, `binary_mask_upsampled.png`, `binary_mask_upsampled_overlay.png`, and a GT overlay when available, alongside the numeric arrays and metadata. Optional SAM2 outputs contain refined binary masks and overlays.

Existing completed frames are skipped. A recorded run cannot silently reuse another probe path, checkpoint override, or different dataset settings: choose a new `--output-dir` for a different configuration, including when checkpoint files are replaced in place. Training outputs and token exports are never modified. Re-running the same command with `--sam2` can reuse the coarse outputs and add refinement.

## Integration

`scripts/evaluate.sh` is a thin entrypoint into the existing experiment coordinator, not a second inference implementation. It may be called by absolute path from a parent project or a Git submodule; relative CLI paths are resolved from the caller's directory, and the working-directory change is confined to the child process. The reusable feature adapters, probe scoring, mask construction, and evaluation remain in `isdynamic/`. Hugging Face publishing is optional tooling and adds no dependency to inference.
