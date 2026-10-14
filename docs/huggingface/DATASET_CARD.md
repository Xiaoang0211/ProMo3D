---
language: en
task_categories:
  - image-segmentation
tags:
  - motion-segmentation
  - frozen-tokens
  - pmd
extra_gated_heading: YouTube-VOS research-use agreement
extra_gated_prompt: "Downloads contain YouTube-VOS RGB images and derived binary masks. You must accept the original terms at https://youtube-vos.org/dataset/term/ and use the data only for non-commercial research. Original ownership and copyright are retained."
extra_gated_fields:
  I accept the original YouTube-VOS terms and non-commercial research restriction: checkbox
extra_gated_button_content: Agree and request access
---

# ProMo3D Training and Validation Assets

[Paper](https://openreview.net/forum?id=bSpkdwCVuy) | [Project page](https://xiaoang0211.github.io/ProMo3D/) | [PMD models](https://huggingface.co/Xiaoang0211/ProMo3D)

## Status

<!-- RELEASE_STATUS -->
**Asset release in preparation.** This initial dataset card does not contain downloadable datasets. Files and checksums will be included with the prepared release.
<!-- END_RELEASE_STATUS -->

## Contents

The source supervision uses PMD's **160 training sequences and shared 80-sequence validation subset**, referred to as YouTube-VOS in the paper and code. These are the exact binary PNG masks used by PMD, not regenerated official instance annotations. Zero denotes static/background and nonzero denotes dynamic. Mask interpretation follows the source supervision and does not label physical velocity.

| Split | Sequences | Mask frames |
|---|---:|---:|
| Train | 160 | 4,933 |
| Validation | 80 | 2,893 |

```text
youtubevos_160vids/train/binary_motion_masks.zip
youtubevos_160vids/train/rgb_images.zip
youtubevos_160vids/train/manifest.json
youtubevos_160vids/validation/binary_motion_masks.zip
youtubevos_160vids/validation/rgb_images.zip
youtubevos_160vids/validation/manifest.json
manifest.json
```

RGB images and binary masks are separate archives for each split. RGB archives contain `JPEGImages/<sequence>/<frame>.jpg`; mask archives contain `Annotations/<sequence>/<frame>.png`. Original file bytes and names are preserved. Each split manifest lists sequence membership, image-mask correspondence, and per-file checksums; the root manifest contains archive checksums. No pre-exported tokens, target-benchmark data, or example videos are included. All six backbones share this source dataset; their probes are stored separately in the model repository.

The page is public. Download access requires a Hugging Face account and agreement to the original research-use terms; requests are approved automatically. This applies to the dataset repository, including both RGB and mask files. PMD probe downloads in the separate model repository are not gated.

## Download and Original Source

After accepting the terms on this page and logging in locally, the reproduction code downloads and verifies both RGB and mask archives:

```bash
bash scripts/download_huggingface.sh --assets data
```

**Source provenance:** the locally retained annotation headers identify training as version 2.0 (2021) and validation as version 3.0 (2022). These correspond to [YouTube-VIS releases](https://youtube-vos.org/dataset/vis/), built on the YouTube-VOS collection. The exact sequence/frame manifests and released bytes are authoritative for reproducing PMD; this release should not be substituted with the full YouTube-VOS 2018 split based on its name. The reproduction release preserves the actual experiment assets, not a reselected or relabeled dataset.

The original RGB images can also be obtained through the [official provider](https://youtube-vos.org/dataset/vis/), following the [terms of use](https://youtube-vos.org/dataset/term/). PMD `train` refers to the retained official `train` source and PMD `validation` to the retained `valid` source. Do not replace the released binary masks with official instance annotations.

Alternatively, select the sequence IDs and image names listed in each manifest from the official `JPEGImages` directories. Extract the RGB and mask archives into the corresponding local split:

```text
datasets/youtubevos_160vids/train/JPEGImages/<sequence>/<frame>.jpg
datasets/youtubevos_160vids/train/Annotations/<sequence>/<frame>.png
datasets/youtubevos_160vids/validation/JPEGImages/<sequence>/<frame>.jpg
datasets/youtubevos_160vids/validation/Annotations/<sequence>/<frame>.png
```

The PMD loaders also recognize the legacy `JEPGImages` spelling. Training-video subsets such as 10/20/40/80 use the reproduction code's documented subset seeds within the 160-sequence training pool, with the same validation set.

## Evaluation Data

DAVIS, SegTrackv2, and FBMS-59 are target benchmarks, not source training data. Download them from their original providers under the applicable terms. Their images and GT masks are not included by this card release. FBMS-59 metrics use annotated frames only; DAVIS void pixels are excluded.

The code's `scripts/evaluate.sh` accepts the downloaded benchmark roots and produces masks, qualitative outputs, and metric tables without requiring source training/validation assets or pre-exported test tokens.

## Licenses and Privacy

Dataset ownership and upstream licenses remain with their original providers. The annotation license and non-commercial research-use terms apply; this release does not grant additional rights to the original videos. Cite the ProMo3D paper, the YouTube-VOS paper, and the relevant YouTube-VIS dataset releases listed by the official provider. No full upstream dataset archives, target GT, private filesystem paths, or training logs are uploaded.
