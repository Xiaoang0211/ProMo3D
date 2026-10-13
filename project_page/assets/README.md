# Paper visuals

`teaser.pdf` is the existing `Teaser_high.pdf` paper figure, unchanged. `teaser.png` is a 2200-pixel-wide render generated with Poppler:

```bash
pdftoppm -singlefile -f 1 -png -scale-to 2200 teaser.pdf teaser
```

The README uses the PNG for browser compatibility; the vector PDF is retained for the project page and publication materials. No third-party template images are included.
