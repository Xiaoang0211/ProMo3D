# Main-paper reproduction

Run commands from the repository root. The main comparison uses native preprocessing, rank-8 low-rank quadratic probes, and the archived non-safeguard protocol. All target benchmarks share the same mask rule, Otsu + 0.2, and frame context of three frames with stride two.

Native preprocessing already includes padding for Pi3X, Pi3, and VGGT. DA3, MapAnything, and DUSt3R use their native preprocessing for the paper results, without the optional `aspect_pad` transform. Only these latter three models have a separate public padding experiment; it is not part of main-paper reproduction.

| Backbone | PMD token groups | Aggregated blocks |
|---|---|---:|
| Pi3X / Pi3 | `inter_frame_block` | 18 |
| DA3 | `inter_frame_block` | 14 |
| DUSt3R | `dec_view1,head_in_view1` | 12 + 4 |
| VGGT (additional) | `inter_frame_block` | 24 |
| MapAnything (additional) | `inter_frame_block` | 8 |

## Training and benchmark tables

```bash
bash scripts/main/run.sh export pi3x pi3 da3 dust3r
bash scripts/main/run.sh train pi3x pi3 da3 dust3r
bash scripts/main/run.sh masks pi3x pi3 da3 dust3r
bash scripts/main/run.sh refine pi3x pi3 da3 dust3r
bash scripts/main/run.sh eval pi3x pi3 da3 dust3r
bash scripts/main/run.sh tables
```

`export` creates sampled token data for training, validation, and token-level test diagnostics. `train` learns one probe per selected block. **`masks` reads images, runs the frozen backbone, and decodes every spatial token; sampled token exports are not used for dense mask inference.** `refine` uses the saved continuous coarse scores to run SAM2. `eval` reports both versions separately on DAVIS2016, DAVIS2017, DAVIS-all, SegTrackv2, and FBMS-59. DAVIS-all masks are reused for the DAVIS sub-splits by the evaluator.

New outputs are:

```text
outputs/tokens/native/<backbone>/data/isdynamic/youtubevos/...
outputs/main/probes/<backbone>/rank_8/
outputs/main/masks/<backbone>/<group>/<benchmark>/<sequence>/...
outputs/main/eval/<backbone>/<group>/<variant>/<benchmark>.json
outputs/main/tables/motion_masks.{csv,json,md}
```

To run a short image-only demonstration, use the mask stage with a fixed subset:

```bash
BENCHMARKS=davis_all SEQUENCE_TAGS_CSV=breakdance MAX_FRAMES=2 \
bash scripts/main/run.sh masks pi3x
```

Use a separate `OUTPUT_ROOT` for demonstrations when you want to keep partial outputs away from complete benchmark runs. Full mask exports include binary masks, score arrays, metadata, and visualizations. On 16 GB GPUs, DA3 defaults to a feature batch of one and a score batch of 32768; the other models use four and 262144. These batching controls do not change frame context.

## Figures and ablations

```bash
# Figure 2: linear, diagonal quadratic, and rank-8 probes across token groups.
bash scripts/main/run.sh layerwise pi3x da3 dust3r
bash scripts/main/run.sh figures --panels layerwise

# Rank-restricted probing and the capacity table.
bash scripts/main/run.sh ranks pi3x --ranks 4 8 16 32 64 128 256
bash scripts/main/run.sh tables --table ranks

# Figure 5(a): low-rank and full-rank probes as scene diversity decreases.
bash scripts/main/run.sh starvation pi3x da3 dust3r --sizes 160 80 40 20 10 --seeds 42 13 7
bash scripts/main/run.sh figures --panels starvation

# Figure 5(b): coarse and refined mask evaluation for each Pi3X subset.
bash scripts/main/run.sh starvation_masks pi3x --sizes 160 80 40 20 10 --seeds 42 13 7
bash scripts/main/run.sh figures pi3x --panels starvation_masks

# Figure 4: probe-induced PCA, singular spectrum, and effective rank.
bash scripts/main/run.sh figures pi3x --panels pca

# Submitted unseen-class qualitative examples (Pi3X).
bash scripts/main/run.sh qualitative
```

**Layer-wise and starvation plots need stored probe/evaluation logs only; they do not need token or mask exports to be loaded when drawing the plot.** The probe-induced PCA panel additionally needs sampled tokens for the plotted split and a trained probe. Its default example is Pi3X inter-frame block 0 on DAVIS-all. Spectral/effective-rank analysis needs trained probes. SAM2-refined starvation metrics must be generated before drawing the corresponding mask panel.

The 160-video point uses seed 42; reduced subsets use 42/13/7 and retain their sequence manifests. The data-starvation plot follows the submission's best-block diagnostic, while downstream PMD always aggregates the predefined blocks. The backbone diagnostic instead averages all selected blocks.

The qualitative command retains the existing example exporter's eight-frame, stride-one context and fixed frames for cheetah, bird of paradise, parachute, worm, koala, and rhino. These image-only examples do not require token export. Benchmark mask metrics use the three-frame, stride-two protocol above.

Use `--dry-run` with any command to view the planned stages without creating artifacts or loading a model. External comparison values in the paper are quoted from their cited sources; the release recomputes PMD results and the runtime comparisons explicitly described in the reported comparisons.
