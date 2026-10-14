import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]


def test_readme_uses_independently_hosted_visuals():
    readme = (ROOT / "README.md").read_text()
    images = re.findall(r'<img[^>]+src="([^"]+)"', readme)
    assert len(images) == 6
    official_neurips_logo = "https://neurips.cc/media/Press/NeurIPS_logo.svg"
    assert images[0] == official_neurips_logo
    assert all(url.startswith("https://xiaoang0211.github.io/ProMo3D/assets/") for url in images[1:])
    assert 'alt="Official NeurIPS logo"' in readme
    assert "project_page/" not in readme


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
    assert "methodological reference, not a runtime dependency" in readme


def test_readme_provides_paper_bibtex_with_camera_ready_authors():
    readme = (ROOT / "README.md").read_text()
    citation = readme.split("## Citation\n", 1)[1].split("## Acknowledgements\n", 1)[0]
    assert "```bibtex\n@inproceedings{zhang2026promo3d," in citation
    assert "{{ProMo3D}: Probing Motion Cues in Frozen {3D} Foundation Models}" in citation
    assert "{Zhang, Xiaoang and Kiray, Mert and Velikova, Yordanka and Biswanath, Manoj and Busam, Benjamin}" in citation
    assert "{Advances in Neural Information Processing Systems}" in citation
    assert "{2026}" in citation
    assert "{https://neurips.cc/virtual/2026/poster/151302}" in citation
