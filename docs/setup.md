# Setup

The public package keeps the validated `isdynamic` import name. Shell entrypoints select the right Python environment for each backbone; they do not require one shared environment containing every third-party project.

## On our current machine

Reuse the installed code, checkpoints, environments, and downloaded datasets:

```bash
cd /home/xiaoang/FF_Reconstruction/promo3d
bash scripts/setup_local.sh /home/xiaoang/FF_Reconstruction/isDynamicProbe
bash scripts/check_setup.sh
```

This creates links under `models/` and `datasets/` and writes `configs/local.env` if it does not exist. Experiment code runs from `promo3d`. Token exports and newly generated results default to `promo3d/outputs/`.

To use existing token exports, set per-model source roots at runtime:

```bash
PI3X_TOKEN_SOURCE=/path/to/pi3x_tokens/data/isdynamic/youtubevos/tokens \
bash scripts/main/run.sh train pi3x
```

The source must contain `train`, `val`, and the named test splits beneath the sequence-sharded `tokens/` root. Prefer a runtime override for archived artifacts. Use `TOKEN_ROOT=/path/to/ssd/promo3d_tokens` when making a new export; it creates `native/<backbone>/...` and `aspect_pad/<backbone>/...` scopes.

## Independent installation

```bash
bash scripts/setup_third_party.sh Pi3 Depth-Anything-3 dust3r sam2
# Additional backbones used in the extended experiments:
bash scripts/setup_third_party.sh vggt map-anything
cp configs/local.env.example configs/local.env
```

The installer checks out the revisions recorded in `configs/third_party.lock.json` and applies the corresponding feature-extraction patches. These patches expose intermediate outputs required by PMD. Install each repository's environment using its own installation instructions, then install this workspace's `requirements.txt` in the relevant environments. Choose a CUDA-enabled PyTorch build appropriate to your GPU. Set `*_PYTHON_BIN` in `configs/local.env` to each environment's interpreter.

Checkpoints are supplied under their original licenses. The defaults are:

| Backbone | Checkpoint location |
|---|---|
| Pi3X | `models/Pi3/ckpt/model_pi3x.safetensors` |
| Pi3 | `models/Pi3/ckpt/model_pi3.safetensors` |
| VGGT | `models/vggt/ckpt/model.pt` |
| DA3 | `models/Depth-Anything-3/src/depth_anything_3/ckpt/DA3-giant-1.1/` |
| MapAnything | `models/map-anything/ckpt/` |
| DUSt3R | `models/dust3r/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth` |
| SAM2 | `models/sam2/checkpoints/sam2.1_hiera_large.pt` |

The lock file contains the upstream repository URLs. Local third-party patches and checkpoint configurations are part of the reproduction protocol, so preserve them when changing environments.

## Datasets

```bash
bash scripts/download_datasets.sh
```

This obtains the 160-training/80-validation source RGB/masks from Hugging Face, plus DAVIS 2017 train/validation at 480p, SegTrackv2, and both FBMS-59 splits. Accept the dataset-page research-use terms and log into the local Hugging Face client first. Existing source subsets are reused. Training-video ablations derive smaller subsets from the 160 videos in code; a separate 10-video download is not required. Official object-instance annotations are not an interchangeable replacement for the supplied binary motion masks. Exact source-release metadata is documented on the dataset card.

YouTube-VOS subset archives may use `train/validation` instead of `train_motion/valid_motion`; the runner builds a local directory view with the expected names. FBMS uses `Trainingset/` and `Testset/`, and only annotated frames contribute to evaluation.

Run `bash scripts/check_setup.sh --models pi3x pi3 vggt da3 mapanything dust3r` after setup. This checks paths and revisions without loading GPU models.
