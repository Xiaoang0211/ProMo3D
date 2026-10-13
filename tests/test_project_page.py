from html.parser import HTMLParser
import json
from pathlib import Path

import pytest

from scripts.build_project_page_assets import frame_dirs

PAGE = Path(__file__).resolve().parents[1] / "project_page"


class AssetLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.paths = []

    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key in {"src", "href", "poster"} and value and not value.startswith(("#", "https://", "http://")):
                self.paths.append(value)


def test_project_page_references_existing_local_assets():
    parser = AssetLinks()
    parser.feed((PAGE / "index.html").read_text())
    assert parser.paths
    assert all((PAGE / path).is_file() for path in parser.paths)


def test_showcase_has_three_complete_sequences_and_four_real_views():
    samples = json.loads((PAGE / "assets/samples.json").read_text())
    provenance = json.loads((PAGE / "assets/provenance.json").read_text())
    assert [sample["id"] for sample in samples] == ["rhino", "koala", "bird_of_paradise"]
    for sample, source in zip(samples, provenance["sequences"]):
        assert sample["frames"] > 80
        assert source["frame_range"] == [0, sample["frames"] - 1]
        assert len(source["blocks"]) == 18
        assert set(sample["views"]) == {"input", "scores", "coarse", "refined"}
        for view in sample["views"].values():
            assert (PAGE / view["video"]).stat().st_size > 0
            assert (PAGE / view["poster"]).stat().st_size > 0


def test_partial_sequence_cannot_be_presented_as_complete(tmp_path):
    for index in (0, 1, 3):
        (tmp_path / "bilinear_upsample" / f"frame_{index:05d}").mkdir(parents=True)
    with pytest.raises(ValueError, match="Incomplete sequence"):
        frame_dirs(tmp_path, 4)


def test_resource_placeholders_do_not_advertise_huggingface_demo():
    html = (PAGE / "index.html").read_text()
    assert "Hugging Face Model Card" in html
    assert "huggingface.co/spaces" not in html
    assert "Hugging Face Demo" not in html
    assert 'href="https://github.com/Xiaoang0211/promo3d"' not in html
