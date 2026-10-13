from html.parser import HTMLParser
from html import unescape
import json
from pathlib import Path
import re

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
    assert [sample["id"] for sample in samples] == ["rhino", "koala", "cheetah"]
    for sample, source in zip(samples, provenance["sequences"]):
        assert sample["frames"] == {"rhino": 90, "koala": 100, "cheetah": 29}[sample["id"]]
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


def test_page_uses_exact_paper_title_and_official_poster_details():
    html = (PAGE / "index.html").read_text()
    title = "ProMo3D: Probing Motion Cues in Frozen 3D Foundation Models"
    heading = re.search(r"<h1>(.*?)</h1>", html, re.DOTALL).group(1)
    assert " ".join(unescape(re.sub(r"<[^>]+>", " ", heading)).split()) == title
    assert f"<title>{title}</title>" in html
    assert '<div class="conference-label">NeurIPS 2026</div>' in html
    assert "Wednesday, 9 December" in html
    assert "10:00-13:00 AEDT (UTC+11)" in html
    assert "ICC Sydney, Hall 1-4" in html
    assert "neurips.cc/virtual/2026/poster/151302" in html


def test_hero_has_no_summary_badges():
    html = (PAGE / "index.html").read_text()
    hero = re.search(r'<section class="hero shell">(.*?)</section>', html, re.DOTALL).group(1)
    assert 'class="facts"' not in hero
    assert "<strong>Rank 8</strong>" not in hero
    assert "target-benchmark transfer" not in hero


def test_representation_analysis_has_three_panels():
    html = (PAGE / "index.html").read_text()
    assert re.findall(r'data-panel="([^"]+)"', html) == ["layerwise", "pca", "starvation"]
    assert "Additional backbones" not in html
    assert 'href="#evidence">Analysis</a>' in html


def test_github_pages_project_path_has_relative_assets():
    html = (PAGE / "index.html").read_text()
    assert '<link rel="canonical" href="https://xiaoang0211.github.io/ProMo3D/">' in html
    assert "https://xiaoang0211.github.io/ProMo3D/assets/teaser.png" in html
    assert (PAGE / ".nojekyll").is_file()
    parser = AssetLinks()
    parser.feed(html)
    assert all(not path.startswith("/") for path in parser.paths)


def test_page_uses_demo_and_reproduce_labels():
    html = (PAGE / "index.html").read_text()
    assert 'href="#sequences">Demo</a>' in html
    assert "<h2>Reproduce</h2>" in html
    assert 'href="#sequences">Sequences</a>' not in html
    assert "<h2>Reproducibility</h2>" not in html
