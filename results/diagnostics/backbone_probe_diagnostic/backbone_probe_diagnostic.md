# Cross-backbone representation diagnostic

Primary diagnostic split: YouTube-VOS validation. Values are source-validation diagnostics and do not use DAVIS, SegTrackv2, or FBMS-59 for backbone selection.

| Backbone | Selected token group | Normalized Gain across selected blocks, mean ± std. | NG min–max | Pairwise compatibility accuracy across selected blocks, mean ± std. | Coarse mean J-M over D17/STv2/FBMS | Rank by Normalized Gain | Rank by coarse J-M |
|---|---|---:|---:|---:|---:|---:|---:|
| pi3 | inter_frame_block (18) | 76.76 ± 0.78 | 75.60–78.01 | 96.39 ± 0.12 | 72.77 | 1 | 2 |
| VGGT | inter_frame_block (24) | 73.60 ± 4.18 | 66.30–78.29 | 95.90 ± 0.65 | 72.31 | 2 | 3 |
| pi3X | inter_frame_block (18) | 72.46 ± 2.06 | 69.05–75.01 | 95.72 ± 0.32 | 73.35 | 3 | 1 |
| MapAnything | inter_frame_block (8) | 67.20 ± 2.10 | 63.54–70.28 | 92.99 ± 0.45 | 66.86 | 4 | 5 |
| DA3 | inter_frame_block (14) | 61.87 ± 6.60 | 44.61–66.71 | 91.91 ± 1.40 | 68.58 | 5 | 4 |
| DUSt3R | dec_view1 + head_in_view1 (16) | 33.73 ± 5.89 | 19.99–40.87 | 85.90 ± 1.25 | 61.59 | 6 | 6 |

Normalized Gain and pairwise compatibility accuracy are reported as percentages. In the stored rank-8 logs, this accuracy is pairwise same/different-motion-state classification accuracy; NG uses the corresponding majority pair-label baseline. A separate unary dynamic-token accuracy is not stored in these rank-8 logs.

| Representation diagnostic | Backbones included | Spearman correlation with coarse mean J-M | Pearson correlation with coarse mean J-M |
|---|---|---:|---:|
| Mean Normalized Gain across selected blocks | all 6 | 0.771 | 0.928 |
| Mean Normalized Gain across selected blocks | 5 multi-view only | 0.600 | 0.802 |
| Mean pairwise compatibility accuracy across selected blocks | all 6 | 0.771 | 0.965 |
| Mean pairwise compatibility accuracy across selected blocks | 5 multi-view only | 0.600 | 0.898 |

The source-validation diagnostic is a useful screening signal, not an exact ranking rule: pi3 has the highest mean validation NG, while pi3X has the highest coarse J-M. For pi3X versus DA3, both the mean and stability support the observed difference: 72.46 vs. 61.87 NG, with across-block population standard deviations of 2.06 vs. 6.60.

The DAVIS-all probe metrics used by Figures 2/7 provide a target-benchmark diagnostic only and are retained in the CSV/JSON outputs; they are not presented as a target-free model-selection criterion.
