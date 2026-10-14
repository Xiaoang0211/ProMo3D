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
    assert "Hugging Face Model Card" not in html
    assert 'href="https://huggingface.co/Xiaoang0211/ProMo3D"' in html
    assert 'href="https://huggingface.co/datasets/Xiaoang0211/ProMo3D-data"' in html
    assert 'href="https://github.com/Xiaoang0211/ProMo3D"' in html
    assert "private repository" not in html
    assert "currently private" not in html
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


def test_probe_induced_subspace_note_is_scoped_to_pca():
    html = (PAGE / "index.html").read_text()
    note = re.search(r'<p id="pca-note".*?</p>', html).group()
    assert 'class="scope-note" hidden' in note
    assert "supervised <strong>probe-induced subspace</strong>" in note
    assert "motion-relevant information for static-dynamic separation" in note
    assert "semantic leakage" not in note
    script = (PAGE / "app.js").read_text()
    assert "document.querySelector('#pca-note').hidden = button.dataset.panel !== 'pca'" in script


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
    assert html.count('<article class="method-stage"') == 4
    assert provenance["method"]["target_frame"] == 70
    assert provenance["method"]["context_frames"] == [68, 72]
    assert provenance["method"]["output_views"] == json.loads((PAGE / "assets/samples.json").read_text())[0]["views"]
    assert provenance["method"]["token_diagram"] == "schematic"
    for frame in (68, 72):
        image_path = PAGE / f"assets/method/rhino-context-{frame}.png"
        assert hashlib.sha256(image_path.read_bytes()).hexdigest() == provenance["method"]["context_sha256"][str(frame)]


def test_input_frames_decrease_in_size_with_depth():
    css = (PAGE / "style.css").read_text()
    for index, width, left in ((1, 80, 10), (2, 90, 5), (3, 100, 0)):
        rule = re.search(rf"\.input-stack img:nth-child\({index}\)\{{([^}}]+)\}}", css).group(1)
        assert f"width:{width}%" in rule
        assert f"left:{left}" in rule
        assert f"z-index:{index}" in rule


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
    assert '<h3 class="stage-title" id="method-backbone-title"><span>01</span> Frozen 3D backbone</h3>' in html
    assert '<span class="stage-note">Input frames</span>' in html
    assert "Low-rank quadratic probe" in html
    assert "U, V:" not in html
    assert "<mtext>stat</mtext>" in html
    assert '<mover accent="true"><mi>p</mi><mo>^</mo></mover>' in html
    assert "The anchor a = 0 lies in Pi3X's padded border and is static by design." in html
    assert "compatibility with the static anchor gives the static score" in html
    assert 'alt="PMD coarse mask supplied to SAM2 refinement"' in html
    assert '<span class="stage-model external-model">SAM2</span>' in html


def test_method_has_no_module_subtitles_branding_or_animation_controls():
    html = (PAGE / "index.html").read_text()
    assert 'class="stage-kind"' not in html
    assert 'class="pipeline"' not in html
    assert 'class="step-detail"' not in html
    assert 'class="method-foot"' not in html
    assert 'class="sam2-logo"' not in html
    assert 'class="sam2-icon"' not in html
    assert "animate-pipeline" not in html
    script = (PAGE / "app.js").read_text()
    assert "animate-pipeline" not in script
    assert "setInterval" not in script
    assert "showStep" not in script


def test_each_method_step_is_explained_in_place():
    html = (PAGE / "index.html").read_text()
    stages = re.findall(r'<article class="method-stage".*?</article>', html, re.DOTALL)
    assert len(stages) == 4
    assert all(stage.count('class="stage-title"') == 1 for stage in stages)
    assert all(stage.count('class="stage-explanation"') == 1 for stage in stages)
    assert "selected intermediate layers, based on the layer-wise probing results" in stages[0]
    assert "token group is fixed per backbone" in stages[0]
    assert "scores are averaged across all its layers" in stages[0]
    assert "feature vector (a token)" not in html
    assert "rank-8 probe" in stages[1]
    assert "160 YouTube-VOS videos" in stages[1]
    assert "predict whether two tokens from the same target frame share the same motion state" in stages[1]
    diagram = re.search(r'<svg class="pair-diagram".*?</svg>', stages[1], re.DOTALL).group()
    assert 'font-size="12">a</tspan>' in diagram
    assert 'font-size="12">0</tspan>' not in diagram
    assert "same thresholding rule across benchmarks" in stages[2]
    assert "<math" not in stages[2]
    assert "region masks generated from PMD scores as SAM2 prompts" in stages[3]


def test_page_uses_consistent_direct_research_wording():
    html = (PAGE / "index.html").read_text()
    assert "low-rank probes trained on 160 YouTube-VOS videos" in html
    assert "Bash scripts cover" in html
    assert "Pre-exported test tokens are not required for mask inference" in html
    assert "Figure and video sources" in html
    for phrase in ("low-rank readouts", "Bash entrypoints", "sampled test exports", "Visual provenance"):
        assert phrase not in html
    script = (PAGE / "app.js").read_text()
    assert "the number of training videos decreases from 160 to 10" in script
    assert "training scene diversity decreases" not in script


def test_page_omits_guest_footnote_and_frame_group_details():
    html = (PAGE / "index.html").read_text()
    assert "guest researcher" not in html
    assert '<sup>1*</sup>' not in html
    assert "Xiaoang Zhang<sup>1</sup>" in html
    assert "Frame groups:" not in html
    assert "stride 2" not in html


def test_page_has_paper_abstract_without_preprocessing_section():
    html = (PAGE / "index.html").read_text()
    abstract = re.search(r'<section id="abstract".*?</section>', html, re.DOTALL).group()
    assert "<h2 id=\"abstract-title\">Abstract</h2>" in abstract
    assert "low-capacity, rank-restricted probes trained with limited source-domain supervision" in abstract
    assert "data-efficient supervised recovery of motion-relevant information" in abstract
    assert html.index('id="abstract"') < html.index('id="sequences"')
    assert "Backbone preprocessing" not in html
    assert 'class="preprocessing"' not in html


def test_semantic_palette_matches_original_teaser_and_css():
    palette = json.loads((PAGE / "assets/palette.json").read_text())
    assert hashlib.sha256((PAGE / "assets" / palette["source"]["file"]).read_bytes()).hexdigest() == palette["source"]["sha256"]
    assert palette["static"] == "#1F77B4"
    assert palette["dynamic"] == "#D62728"
    css = (PAGE / "style.css").read_text()
    for key in ("static", "dynamic", "backbone_fill"):
        assert f"--{key.replace('_', '-')}:{palette[key]}" in css
    for index, color in enumerate(palette["raw_tokens"], 1):
        assert f"--token-purple-{index}:{color}" in css
    assert f'--probe-bg:{palette["probe"]["background"]}' in css
    assert f'--probe-fill:{palette["probe"]["fill"]}' in css
    assert f'--probe-border:{palette["probe"]["border"]}' in css


def test_unknown_tokens_are_purple_and_known_static_anchor_is_blue():
    html = (PAGE / "index.html").read_text()
    assert 'class="raw-tokens"' in html
    diagram = re.search(r'<svg class="pair-diagram".*?</svg>', html, re.DOTALL).group()
    assert all(f'class="feature-channel-{i}"' in diagram for i in range(1, 6))
    assert "#d5ddcb" not in diagram
    assert 'fill="var(--static)"' in diagram
    anchor = re.search(r'<g class="anchor-token".*?</g>', diagram, re.DOTALL).group()
    assert anchor.count('<rect ') == 5
    assert anchor.count('width="14"') == 5
    assert 'width="92" height="56" rx="2" fill="var(--static)"' not in diagram
    assert "Anchor token" in diagram
    assert "Token being scored" in diagram
    assert "Both are spatial tokens from the same target frame" in diagram


def test_project_page_abstract_matches_camera_ready_when_source_is_available():
    source = PAGE.parent.parent / "isDynamicProbe/3D_Pretext/neurips_2026.tex"
    if not source.is_file():
        pytest.skip("Camera-ready Overleaf source is maintained outside the public code repository")
    tex = source.read_text()
    abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.DOTALL).group(1).split(r"\par")[0]
    abstract = abstract.replace(r"$\pi^3$X", "\u03c03X").replace("--", "\u2013")
    section = re.search(r'<section id="abstract".*?</section>', (PAGE / "index.html").read_text(), re.DOTALL).group()
    paragraph = re.search(r"<p>(.*?)</p>", section, re.DOTALL).group(1)
    rendered = unescape(re.sub(r"<[^>]+>", "", paragraph))
    assert " ".join(abstract.split()) == " ".join(rendered.split())


def test_method_probe_does_not_repeat_static_dynamic_color_legend():
    html = (PAGE / "index.html").read_text()
    stage = re.search(
        r'<article class="method-stage" aria-labelledby="method-probe-title">.*?</article>',
        html,
        re.DOTALL,
    ).group()
    assert 'class="motion-key"' not in stage
    assert 'class="state-static"' not in stage
    assert 'class="state-dynamic"' not in stage
    assert '<math display="block"' in stage


def test_resource_icons_are_local_and_shared_with_github_readme():
    html = (PAGE / "index.html").read_text()
    resources = re.search(r'<div class="resources".*?</div>', html, re.DOTALL).group()
    assert resources.count('class="resource-icon"') == 5
    readme = (PAGE.parent / "README.md").read_text()
    assert "Hugging Face Model Card" not in readme
    assert "Hugging Face Models & Data" in readme
    icons = re.findall(r'src="(project_page/assets/icons/[^"]+)"', readme)
    assert len(icons) == 5
    assert all((PAGE.parent / icon).is_file() for icon in icons)
    assert "model cards" in readme
    assert "training/validation assets" in readme
