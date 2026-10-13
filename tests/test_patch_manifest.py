import json
from pathlib import Path


def test_pinned_feature_patches_include_da3_helper():
    root = Path(__file__).resolve().parents[1]
    lock = json.loads((root / "configs/third_party.lock.json").read_text())
    for entry in lock.values():
        assert len(entry["commit"]) == 40
        if entry.get("patch"):
            assert (root / entry["patch"]).is_file()
    patch = (root / lock["Depth-Anything-3"]["patch"]).read_text()
    assert "from depth_anything_3.model.utils.attention_recorder import VisionAttentionRecorder" in patch
    assert "+++ b/src/depth_anything_3/model/utils/attention_recorder.py" in patch
