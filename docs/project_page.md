# Project-page preparation

The project page can live in `project_page/`, alongside the public reproduction repository. Use the main method name ProMo3D for the paper and PMD for the probing-to-mask pipeline.

## Reference layout

[UnReflectAnything](https://alberto-rota.github.io/UnReflectAnything/) is a suitable reference for a visual-first research page: paper/resource buttons, an interactive results gallery, method illustrations, and citation. Its website source is on the [`page` branch](https://github.com/alberto-rota/UnReflectAnything/tree/page), inspected at commit `2402d2d0c4e63cbf9ef709dbf6d64a1c14542bbc`. The upstream repository provides an [MIT license](https://github.com/alberto-rota/UnReflectAnything/blob/main/LICENSE). Preserve its copyright and permission notice if reusing source code; its paper figures and branding are not ProMo3D assets.

The implementation uses our own static HTML/CSS/JavaScript and paper assets. It shows synchronized input, dynamic scores, PMD coarse overlays, and PMD + SAM2 overlays as distinct views. No source code, logos, result images, or publication badges were copied from the reference page.

## Paper resources

- Public project page: https://xiaoang0211.github.io/ProMo3D/
- Website-only repository: https://github.com/Xiaoang0211/Xiaoang0211.github.io
- NeurIPS: https://neurips.cc/virtual/2026/poster/151302
- OpenReview: https://openreview.net/forum?id=bSpkdwCVuy
- arXiv: reserved until the paper record is available.
- Hugging Face Model Card: reserve the model repository URL for checkpoints and usage information.

Unavailable resources are labeled "coming soon", not linked to invented URLs. The code repository is private; the public page does not advertise its private URL as a usable download. The README and project page should use the same finalized paper/model links.

The official paper-specific poster schedule was checked on 13 October 2026: Wednesday, 9 December 2026, 10:00-13:00 AEDT (UTC+11), Hall 1-4. The conference venue is ICC Sydney, Australia. Only the project page shows the session details; the repository README retains the NeurIPS 2026 paper and project-page links.

Suggested page order:

1. Paper title, authors/affiliations, paper/code links, and the teaser.
2. Short explanation of frozen tokens, pairwise motion compatibility, and coarse masks.
3. Coarse mask video examples with optional SAM2 refinement shown separately.
4. Layer-wise decodability, probe-induced PCA, and scene-diversity plots.
5. Main and additional experiment reproduction links and citation.

Use `outputs/main/figures/` and the full mask-stage visualizations as regenerated assets. Keep the submitted values and supplemental values identifiable when selecting table/figure assets. No private OpenReview comments or confidential AC messages should be published with the page.

The README teaser is `project_page/assets/teaser.png`, rendered from the existing paper figure without changing its content; `teaser.pdf` retains the vector original. Both can be shared with the project page.

Author metadata and affiliations were checked against the camera-ready PDF and official NeurIPS listing. The page includes a BibTeX citation without inventing volume or page numbers. The PMD license decision and resource release URLs remain to be finalized before code release.

## Build and access

Run `bash scripts/build_project_page.sh` to prepare `project_page/dist/`. The optional `--assets --archive_outputs /path/to/outputs` rebuilds twelve videos for rhino, koala, and cheetah, and three analysis figures without new inference. It validates that every source frame has a stored prediction and checks each encoded video's frame count. The clips use 12 fps for visualization; they do not report inference latency.

The project page is deployed independently of the private GitHub code repository. The public `Xiaoang0211.github.io` repository contains the generated page in `ProMo3D/`; its Pages source is `main`, directory `/`. Only `dist/` is copied there: no datasets, weights, probe exports, raw experiment logs, local environment configuration, or confidential review material.

Run `bash scripts/publish_project_page.sh` to build and update the public page. The script changes only the website repository's `ProMo3D/` directory and never force-pushes. The root `.nojekyll` keeps the HTML/CSS/JavaScript and assets unchanged. The old Sites manifest is retained for the previous deployment, but is not included in GitHub Pages. Credentials are not stored in either repository.
