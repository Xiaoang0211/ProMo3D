import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]


def test_readme_introduces_the_research_before_setup_and_reproduction():
    readme = (ROOT / "README.md").read_text()
    introduction = readme.split("## Method\n", 1)[0]
    assert "Official implementation of **ProMo3D**" in introduction
    assert "NeurIPS 2026" in introduction
    assert "low-capacity supervised probing" in introduction
    assert "probe-induced representation" in introduction
    assert "Probing-based Motion Distillation (PMD)" in introduction
    assert "reproduc" not in introduction.lower()
    assert "private project-page source" not in readme
    assert readme.index("## Method\n") < readme.index("## Setup\n")
    assert readme.index("## Pretrained Inference\n") < readme.index("## Reproduction\n")


def test_readme_uses_vector_resource_logos_and_independently_hosted_teaser():
    readme = (ROOT / "README.md").read_text()
    images = re.findall(r'<img[^>]+src="([^"]+)"', readme)
    assert len(images) == 6
    assert images[:5] == [f"assets/links/{name}.svg" for name in ("paper", "project", "arxiv", "models", "data")]
    assert all(url.endswith(".svg") for url in images[:5])
    assert images[-1] == "https://xiaoang0211.github.io/ProMo3D/assets/teaser.png"
    assert "project_page/" not in readme


def test_resource_links_have_uniform_vector_layout_and_no_broken_arxiv_link():
    ns = "{http://www.w3.org/2000/svg}"
    for name in ("paper", "project", "arxiv", "models", "data"):
        icon = ET.parse(ROOT / f"assets/links/{name}.svg").getroot()
        assert icon.attrib["viewBox"] == "0 0 132 88"
        assert icon.find(ns + "title") is not None
        assert icon.find(ns + "svg") is not None
        assert icon.findall(".//" + ns + "path")
        assert icon.findall(".//" + ns + "image") == []
        assert icon.findall(".//" + ns + "text") == []
    readme = (ROOT / "README.md").read_text()
    links = re.findall(r'<a href="([^"]+)">(<img[^>]+>)</a>', readme)
    assert len(links) == 4
    assert not any("links/arxiv.svg" in image for _, image in links)


def test_promo3d_wordmark_uses_outlined_vectors_and_project_colors():
    logo = ET.parse(ROOT / "assets/logos/promo3d.svg").getroot()
    ns = "{http://www.w3.org/2000/svg}"
    assert logo.tag == ns + "svg"
    paths = logo.findall(".//" + ns + "path")
    assert len(paths) == 7
    assert [path.attrib["fill"] for path in paths] == ["#252525"] * 5 + ["#1F77B4"] * 2
    assert logo.findall(".//" + ns + "image") == []
    assert logo.findall(".//" + ns + "text") == []


def test_public_assets_contain_only_images_grouped_by_type():
    assets = ROOT / "assets"
    assert {path.name for path in assets.iterdir()} == {"logos", "links"}
    for path in assets.rglob("*"):
        if path.is_file():
            assert path.suffix.lower() in {".svg", ".png", ".jpg", ".jpeg", ".webp", ".avif", ".gif", ".pdf"}


def test_readme_local_documentation_and_image_links_exist():
    readme = (ROOT / "README.md").read_text()
    links = re.findall(r'\[[^\]]+\]\(([^)]+)\)|<img[^>]+src="([^"]+)"', readme)
    for markdown_link, image in links:
        target = (markdown_link or image).split("#", 1)[0]
        if target and not target.startswith(("https://", "http://")):
            assert (ROOT / target).exists(), target


def test_original_code_has_mit_license_with_third_party_scope_separated():
    license_text = (ROOT / "LICENSE").read_text()
    assert license_text.startswith("MIT License\n")
    assert "Copyright (c) 2026 ProMo3D Authors" in license_text
    assert "Permission is hereby granted, free of charge" in license_text
    assert 'THE SOFTWARE IS PROVIDED "AS IS"' in license_text
    readme = (ROOT / "README.md").read_text()
    assert "[MIT License](LICENSE)" in readme
    assert "does not relicense assets downloaded from Hugging Face" in readme


def test_acknowledgements_cover_pinned_implementations_and_probing_reference():
    readme = (ROOT / "README.md").read_text()
    lock = json.loads((ROOT / "configs/third_party.lock.json").read_text())
    for entry in lock.values():
        assert entry["url"].removesuffix(".git").rstrip("/") in readme
    assert "https://github.com/liyihao0302/vit-object-binding" in readme
    assert "IsSameObject" in readme


def test_readme_provides_paper_bibtex_with_camera_ready_authors():
    readme = (ROOT / "README.md").read_text()
    citation = readme.split("## Citation\n", 1)[1].split("## Acknowledgements\n", 1)[0]
    assert "```bibtex\n@inproceedings{zhang2026promo3d," in citation
    assert "{{ProMo3D}: Probing Motion Cues in Frozen {3D} Foundation Models}" in citation
    assert "{Zhang, Xiaoang and Kiray, Mert and Velikova, Yordanka and Biswanath, Manoj and Busam, Benjamin}" in citation
    assert "{Advances in Neural Information Processing Systems}" in citation
    assert "{2026}" in citation
    assert "{https://neurips.cc/virtual/2026/poster/151302}" in citation
