---
language: en
tags:
  - motion-segmentation
  - frozen-backbones
  - 3d-vision
  - low-rank-probing
  - neurips-2026
---

# ProMo3D: Probing Motion Cues in Frozen 3D Foundation Models

**NeurIPS 2026** | [Paper](https://openreview.net/forum?id=bSpkdwCVuy) | [Project page](https://xiaoang0211.github.io/ProMo3D/)

Xiaoang Zhang, Mert Kiray, Yordanka Velikova, Manoj Biswanath, Benjamin Busam

Technical University of Munich (TUM) | Munich Center for Machine Learning (MCML) | Obsphera

<img src="https://xiaoang0211.github.io/ProMo3D/assets/teaser.png" alt="PMD pipeline from frozen 3D tokens to coarse motion masks and optional SAM2 refinement">

## Release Status

<!-- RELEASE_STATUS -->
This page documents the PMD model release. **Probe checkpoints and the public code release are in preparation.** This initial page does not contain weights. Do not interpret this documentation as an available pretrained checkpoint download.
<!-- END_RELEASE_STATUS -->

Training/validation RGB images and binary motion masks are released separately in [ProMo3D-data](https://huggingface.co/datasets/Xiaoang0211/ProMo3D-data), with agreement to the upstream research-use terms required for downloads. Backbone weights are obtained from their official sources.

## Checkpoint Layout

Each backbone has its own directory. Rank-8 low-rank probes are supplied for all available trained block groups, including the groups used for dense PMD masks. Every block directory contains the original, byte-identical weights-only `best_probe.pt` and a portable `config.json`; `manifest.json` records file checksums and the original checkpoint/configuration hashes. Training logs, optimizer states, source image data, and machine-specific paths are excluded.

```text
pi3x/rank_8/pair_lowrank_quadratic/<group>/<block>/
pi3/rank_8/pair_lowrank_quadratic/<group>/<block>/
vggt/rank_8/pair_lowrank_quadratic/<group>/<block>/
da3/rank_8/pair_lowrank_quadratic/<group>/<block>/
mapanything/rank_8/pair_lowrank_quadratic/<group>/<block>/
dust3r/rank_8/pair_lowrank_quadratic/<group>/<block>/
```

Use the original backbone/checkpoint configuration, not a replacement model variant. Official sources: [Pi3X and Pi3](https://github.com/yyfz/Pi3), [VGGT](https://github.com/facebookresearch/vggt), [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3), [MapAnything](https://github.com/facebookresearch/map-anything), and [DUSt3R](https://github.com/naver/dust3r). The paper uses DA3-GIANT-1.1, the original non-commercial VGGT-1B, and the non-commercial MapAnything weights. Their upstream licenses apply.

## Method

Probing-based Motion Distillation (PMD) recovers motion-relevant information from frozen 3D foundation-model tokens with a low-rank quadratic probe trained on 160 YouTube-VOS videos. A probe predicts whether two tokens from the same target frame share the same motion state. At inference, the fixed anchor token `h_a` and the token `h_i` being scored are ordinary spatial tokens with different roles, not attention query/key projections.

For the padded Pi3X configuration, anchor index `a = 0` is static by construction. Compatibility with that anchor gives the static score, and its complement gives the dynamic score. Scores are averaged over the fixed selected block group, bilinearly upsampled, and binarized. SAM2 is optional external refinement; coarse and refined results are reported separately.

Supported release configurations are **Pi3X, Pi3, VGGT, DA3, MapAnything, and DUSt3R**. The paper uses native backbone preprocessing: Pi3X/Pi3/VGGT include their native padding; the other three use native preprocessing without the additional aspect-padding experiment. The Pi3X rank-8 probe has 294,930 trained parameters across its 18 aggregated blocks. Frozen backbone parameters are not included in that count.

These are the original paper probes trained from the original sampled-token exports, without forced token-0 export or safeguard training. Native padding in Pi3X/Pi3/VGGT is retained and must not be confused with those later ablations. Released checkpoints are matched against the archived paper metrics before packaging.

## Coarse Results

Each cell reports **J-Mean / J-Recall (%)**, without SAM2. PMD uses no target-benchmark training or fine-tuning.

| Backbone | DAVIS2017 | SegTrackv2 | FBMS-59 |
|---|---:|---:|---:|
| Pi3X | 73.7 / 91.0 | 72.9 / 94.0 | 73.5 / 88.8 |
| Pi3 | 72.7 / 89.9 | 72.9 / 92.9 | 72.6 / 83.4 |
| VGGT | 72.6 / 89.9 | 70.8 / 84.7 | 73.5 / 85.9 |
| DA3 | 69.6 / 85.8 | 66.3 / 80.3 | 69.8 / 81.4 |
| MapAnything | 69.2 / 86.0 | 62.3 / 75.7 | 69.1 / 79.9 |
| DUSt3R | 62.2 / 70.0 | 58.6 / 77.3 | 64.0 / 74.5 |

Source: camera-ready main/additional-backbone tables. J-Recall counts frames with Jaccard score above 0.5; it is not pixel recall. Metrics are sequence-macro averages and FBMS-59 uses only annotated frames.

## Direct Mask Inference and Evaluation

The reproduction repository provides `scripts/download_huggingface.sh` and `scripts/evaluate.sh`. These download the PMD probes and run existing dense inference, qualitative-output writing, and evaluation without training or token export. The selected backbone environment/checkpoint is still required. A Pi3X example is:

```bash
bash scripts/download_huggingface.sh --assets models --models pi3x

bash scripts/evaluate.sh pi3x \
  --probe-root outputs/pretrained/pi3x/rank_8/pair_lowrank_quadratic \
  --davis-root /path/to/DAVIS \
  --segtrack-root /path/to/SegTrackv2 \
  --fbms-root /path/to/FBMS-59 \
  --output-dir outputs/predictions/pi3x_native
```

Add `--sam2` to generate refined masks and evaluate coarse/refined outputs separately. Add `--dry-run` to inspect commands. Outputs include input images, score heatmaps, masks, overlays, evaluation JSONs, and a combined J-Mean/J-Recall CSV. Different inference settings require a new output directory. The fixed image-inference context is 3 frames with stride 2.

Sampled training/validation token exports are not full spatial grids and are not a substitute for dense image inference.

## Intended Use and Limitations

PMD is a research method for supervised motion decodability and benchmark motion-mask prediction. Compactness refers to the supervised probe-induced representation. Semantic leakage can occur, and annotation-defined foreground targets need not equal instantaneous physical motion. The fixed-reference assumption is not guaranteed for every native non-padded configuration. Fully uniform same-state scenes do not establish absolute physical motion from pairwise compatibility alone.

The method does not estimate object identities, velocities, or trajectories. Deployment for monitoring people should respect consent, privacy, and applicable dataset restrictions. It is not validated as a safety-critical perception system.

## Licensing

Licenses for the trained probes and training/validation assets will be stated with those artifacts. Third-party backbones, checkpoints, and datasets retain their original licenses. This page does not grant redistribution rights to those third-party assets.
