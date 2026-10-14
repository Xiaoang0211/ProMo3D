import json
import os
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest

from isdynamic.viz.evaluate_motion_masks import db_eval_iou, db_statistics
from isdynamic.viz.motion_mask_utils import binary_mask_from_score, otsu_threshold, save_motion_mask_outputs
from isdynamic.viz.scoremap_utils import average_block_maps, upsample_patch_map


def test_bilinear_then_strict_threshold():
    tokens = np.array([[0, 1], [0, 1]], dtype=np.float32)
    scores = upsample_patch_map(tokens, (4, 4))
    np.testing.assert_allclose(scores, np.tile([0, 0.25, 0.75, 1], (4, 1)))
    mask, threshold = binary_mask_from_score(scores, 0.5)
    np.testing.assert_array_equal(mask, np.tile([0, 0, 1, 1], (4, 1)))
    assert threshold == 0.5 and mask.dtype == np.uint8
    boundary, _ = binary_mask_from_score(np.array([[0.49, 0.50, 0.51]]), 0.5)
    np.testing.assert_array_equal(boundary, [[0, 0, 1]])


def test_otsu_degenerate_and_nonfinite_scores():
    assert otsu_threshold(np.array([np.nan, np.inf])) == 0.5
    mask, threshold = binary_mask_from_score(np.full((2, 2), 0.25), 0.5,
                                           threshold_mode="otsu", threshold_offset=0.2)
    assert threshold == pytest.approx(0.45)
    assert not mask.any()


def test_block_aggregation_averages_every_selected_block():
    maps = {"inter_frame_block_0": np.zeros((2, 2)),
            "inter_frame_block_1": np.ones((2, 2)),
            "inter_frame_block_2": np.ones((2, 2))}
    np.testing.assert_allclose(average_block_maps(maps), 2/3)


def test_iou_and_j_recall_conventions():
    gt = np.array([[1, 1], [0, 0]])
    pred = np.array([[1, 0], [1, 0]])
    assert db_eval_iou(gt, pred) == pytest.approx(1/3)
    assert db_eval_iou(np.zeros((2, 2)), np.zeros((2, 2))) == 1
    stats = db_statistics([0.5, 0.5, 0.6, 0.7])
    assert stats["M"] == pytest.approx(0.575)
    assert stats["R"] == 0.5


@pytest.mark.parametrize("benchmark", ["davis_all", "davis2017"])
def test_png_masks_evaluate_without_token_exports(tmp_path, benchmark):
    davis, masks = tmp_path / "DAVIS", tmp_path / "masks"
    prediction = masks / "pi3x/inter_frame_block/davis_all/toy/bilinear_upsample"
    for folder in (davis / "JPEGImages/480p/toy", davis / "Annotations/480p/toy",
                   davis / "ImageSets/2017", davis / "ImageSets/2016"):
        folder.mkdir(parents=True)
    (davis / "ImageSets/2017/train.txt").write_text("")
    (davis / "ImageSets/2017/val.txt").write_text("toy\n")
    (davis / "ImageSets/2016/val.txt").write_text("toy\n")
    gt = np.zeros((8, 8), dtype=np.uint8)
    gt[2:6, 2:6] = 1
    for frame in range(4):
        Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(davis / f"JPEGImages/480p/toy/{frame:05d}.jpg")
        Image.fromarray(gt).save(davis / f"Annotations/480p/toy/{frame:05d}.png")
        dest = prediction / f"frame_{frame:05d}"
        dest.mkdir(parents=True)
        Image.fromarray(255 * gt).save(dest / "binary_mask_upsampled.png")
    output = tmp_path / "metrics.json"
    process = subprocess.run([sys.executable, "-m", "isdynamic.viz.evaluate_motion_masks", "--model", "pi3x",
                    "--group", "inter_frame_block", "--benchmark", benchmark,
                    "--mask_root", str(masks), "--davis_root", str(davis), "--output_json", str(output)],
                   capture_output=True, text=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    assert process.returncode == 0, process.stderr
    result = json.loads(output.read_text())
    assert result["metrics"]["J"]["M"] == 1
    assert result["metrics"]["J"]["R"] == 1
    assert result["num_frames"] == 4


def test_full_mask_outputs_include_qualitative_views(tmp_path, monkeypatch):
    monkeypatch.setenv("ISDYNAMIC_MASK_OUTPUT_MODE", "full")
    scores = np.array([[0, 1], [0, 1]], dtype=np.float32)
    image = np.full((4, 4, 3), 128, dtype=np.uint8)
    gt = np.tile([0, 0, 1, 1], (4, 1)).astype(np.uint8)
    save_motion_mask_outputs(tmp_path, scores, (4, 4), 0.5, "fixed", False, {},
                             image_np=image, gt_mask_np=gt)
    for filename in ("input_image.png", "avg_score_upsampled_heatmap.png", "binary_mask_upsampled.npy",
                     "binary_mask_upsampled.png", "binary_mask_upsampled_overlay.png", "binary_mask_gt_overlay.png"):
        assert (tmp_path / filename).is_file(), filename
    np.testing.assert_array_equal(np.load(tmp_path / "binary_mask_upsampled.npy"), gt)
    assert json.loads((tmp_path / "meta.json").read_text())["output_mode"] == "full"
