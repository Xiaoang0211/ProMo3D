# Hugging Face Release

Two repositories: `Xiaoang0211/ProMo3D` for the six paper backbones' PMD probes, and `Xiaoang0211/ProMo3D-data` for the 160-training/80-validation RGB images and binary motion masks. The model repository is public. The dataset page is public, with file downloads gated by agreement to the upstream non-commercial research terms. Backbone checkpoints are obtained from their official providers. No tokens, benchmark data, example videos, or hosted application are part of this release.

## Prepare

The archived source must be the original paper `outputs/`, not `outputs_anchor0` or another later experiment. The preparer reuses the source layout recorded by `isdynamic/release/import_results.py`, checks every run against `results/reference/main/probes`, rejects safeguard/residualization variants, and preserves checkpoint bytes. Pi3X/Pi3/VGGT keep native padding; DA3/MapAnything/DUSt3R do not use the additional padding experiment.

```bash
bash scripts/prepare_huggingface.sh \
  --outputs /path/to/original/paper/outputs \
  --youtubevos-root datasets/youtubevos_160vids \
  --destination outputs/releases/huggingface
```

The destination must be new. Preparation writes only its own staging/release directory and never modifies source checkpoints, exports, or masks. It verifies binary mask values, disjoint split membership, and the expected 160/80 sequence counts.

```text
outputs/releases/huggingface/
  models/<backbone>/rank_8/pair_lowrank_quadratic/<group>/<block>/
  models/manifest.json
  models/README.md
  dataset/youtubevos_160vids/train/{rgb_images.zip,binary_motion_masks.zip,manifest.json}
  dataset/youtubevos_160vids/validation/{rgb_images.zip,binary_motion_masks.zip,manifest.json}
  dataset/manifest.json
  dataset/README.md
```

## Publish

Install the optional publishing dependency in a suitable environment, authenticate locally, and preview the operation:

```bash
python -m pip install 'huggingface_hub>=0.27,<1.0'
python -c 'from huggingface_hub import login; login()'
bash scripts/publish_huggingface.sh --release-dir outputs/releases/huggingface --dry-run
```

Browser login and API/CLI authentication are separate. Enter an appropriate Hugging Face token only into the local terminal prompt; never into source files or chat. Then publish:

```bash
bash scripts/publish_huggingface.sh \
  --namespace Xiaoang0211 --release-dir outputs/releases/huggingface
```

The publisher rechecks checksums before remote operations, enables and verifies the dataset access agreement before uploading any RGB data, and uploads only manifest-listed files and the reviewed cards. It does not upload an entire workspace. Omitting `--release-dir` publishes documentation only and explicitly leaves artifact availability as pending.

## Download

```bash
bash scripts/download_huggingface.sh --assets models --models pi3x
bash scripts/download_huggingface.sh --assets data
```

The first command needs no dataset access agreement. For the second, accept the terms on the dataset page and authenticate locally. Downloads are pinned to one repository revision, checksummed, and never overwrite different existing files. Probes default to `outputs/pretrained`; RGB/masks default to `datasets/youtubevos_160vids`. An existing linked dataset is reused by the normal dataset setup script; the dedicated downloader requires a fresh `--data-root` instead of writing through that link.

Use `HUB_PYTHON_BIN=/path/to/python` when dependencies are installed in another environment. Preparation additionally needs the normal CPU probe dependencies (PyTorch, NumPy, Pillow). Update GitHub/project-page model/data links only after verifying the actual public release. The GitHub research repository is public at https://github.com/Xiaoang0211/ProMo3D.
