from html.parser import HTMLParser
from html import unescape
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

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
    assert '>Layer-wise probing</button>' in html
    assert '>Across layers</button>' not in html
    assert "Layer-wise probing, PCA of the probe-induced representation" not in html


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


def test_method_figure_uses_matched_real_outputs():
    html = (PAGE / "index.html").read_text()
    provenance = json.loads((PAGE / "assets/provenance.json").read_text())
    assert re.findall(r'data-stage="([^"]+)"', html) == ["0", "1", "2", "3"]
    assert provenance["method"]["target_frame"] == 70
    assert provenance["method"]["context_frames"] == [68, 72]
    assert provenance["method"]["output_views"] == json.loads((PAGE / "assets/samples.json").read_text())[0]["views"]
    assert provenance["method"]["token_diagram"] == "schematic"
    for frame in (68, 72):
        image_path = PAGE / f"assets/method/rhino-context-{frame}.png"
        assert hashlib.sha256(image_path.read_bytes()).hexdigest() == provenance["method"]["context_sha256"][str(frame)]


def test_static_build_versions_changed_assets(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    build = scripts / "build_project_page.sh"
    shutil.copy2(PAGE.parent / "scripts/build_project_page.sh", build)
    page = tmp_path / "project_page"
    (page / "assets").mkdir(parents=True)
    (page / ".nojekyll").touch()
    (page / "index.html").write_text('<link href="style.css"><script src="app.js"></script>')
    (page / "style.css").write_text("body{}")
    for script in ("const step=0;", "const step=1;"):
        (page / "app.js").write_text(script)
        subprocess.run(["bash", str(build)], check=True, capture_output=True)
        html = (page / "dist/index.html").read_text()
        for name in ("style.css", "app.js"):
            version = hashlib.sha256((page / name).read_bytes()).hexdigest()[:12]
            assert f"{name}?v={version}" in html


def test_method_figure_matches_paper_score_notation_and_refinement_input():
    html = (PAGE / "index.html").read_text()
    assert "Target frame + context frames" in html
    assert "Target frame + two context frames" not in html
    assert "Low-rank quadratic probe" in html
    assert "U, V:" not in html
    assert "<mtext>stat</mtext>" in html and "<mtext>dyn</mtext>" in html
    assert '<mover accent="true"><mi>p</mi><mo>^</mo></mover>' in html
    assert "With padding, token 0 (a = 0) is static by design." in html
    assert "Compatibility with token 0 gives the static score." in html
    assert 'alt="PMD coarse mask supplied to SAM2 refinement"' in html
    assert 'class="sam2-icon"' in html


def test_page_has_paper_abstract_without_preprocessing_section():
    html = (PAGE / "index.html").read_text()
    abstract = re.search(r'<section id="abstract".*?</section>', html, re.DOTALL).group()
    assert "<h2 id=\"abstract-title\">Abstract</h2>" in abstract
    assert "low-capacity, rank-restricted probes trained with limited source-domain supervision" in abstract
    assert "data-efficient supervised recovery of motion-relevant information" in abstract
    assert html.index('id="abstract"') < html.index('id="sequences"')
    assert "Backbone preprocessing" not in html
    assert 'class="preprocessing"' not in html
