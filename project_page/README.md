# Project page

Static ProMo3D project page, with our own HTML/CSS/JavaScript and a visual-first layout inspired by [UnReflectAnything](https://alberto-rota.github.io/UnReflectAnything/).

Public page: https://xiaoang0211.github.io/ProMo3D/

`assets/teaser.png` is shared with the repository README. `assets/videos/` contains all-frame input, motion-score, coarse-overlay, and SAM2-overlay clips for rhino, koala, and cheetah. `assets/provenance.json` records source outputs and exact frame coverage. No online inference is performed by the page.

Build the static page with:

```bash
bash scripts/build_project_page.sh
```

Rebuild visual assets from existing PMD outputs with:

```bash
bash scripts/build_project_page.sh --assets --archive_outputs /path/to/outputs
```

Run these commands from the repository root. Asset rebuilding needs FFmpeg, Poppler, the corresponding benchmark images in `datasets/`, and complete stored Pi3X outputs, but does not run a backbone. The generated `dist/` directory contains only the static page and visual assets. Deployment does not upload the private PMD codebase.

Publish updates with:

```bash
bash scripts/publish_project_page.sh
```

This updates only `ProMo3D/` in the public website-only repository, `Xiaoang0211/Xiaoang0211.github.io`. The research repository remains private. GitHub Pages publishes its `main` branch; no custom domain is required.

The page supports synchronized playback, timeline seeking, playback speed, result-view switching, analysis tabs, enlarged figures, and copying reproduction commands or BibTeX. The four-step method figure explains each stage in place, without duplicate controls. It respects reduced-motion preferences and does not auto-play videos.
