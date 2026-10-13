# Cross-backbone representation diagnostic

The representation diagnostic and source mask metrics use the same 80-video YouTube-VOS validation split. No target benchmark is used for backbone selection.

| Backbone | Selected token group | YT-VOS-val NG, mean ± std. | YT-VOS-val coarse J-M/J-R | Target coarse mean J-M/J-R |
|---|---|---:|---:|---:|
| pi3 | inter_frame_block (18) | 76.76 ± 0.78 | 73.08/88.86 | 72.77/88.76 |
| VGGT | inter_frame_block (24) | 73.60 ± 4.18 | 72.72/88.92 | 72.31/86.82 |
| pi3X | inter_frame_block (18) | 72.46 ± 2.06 | 71.95/88.50 | 73.35/91.28 |
| MapAnything | inter_frame_block (8) | 67.20 ± 2.10 | 67.34/81.19 | 66.86/80.50 |
| DA3 | inter_frame_block (14) | 61.87 ± 6.60 | 64.41/78.42 | 68.58/82.52 |
| DUSt3R | dec_view1 + head_in_view1 (16) | 33.73 ± 5.89 | 56.18/65.62 | 61.59/73.92 |

Normalized Gain is computed as (pair accuracy - majority pair-label baseline) / (1 - majority pair-label baseline), using pairwise same/different-motion-state classification. Source-validation coarse J-M is the primary operational selection metric; J-R is secondary because it thresholds each frame at J > 0.5.

| Diagnostic | Diagnostic split | Outcome | Outcome split | Backbones | Spearman | Pearson |
|---|---|---|---|---|---:|---:|
| Mean Normalized Gain | youtubevos_val | Three-benchmark coarse mean J-M | DAVIS17/SegTrackv2/FBMS-59 | all 6 | 0.771 | 0.928 |
| Mean Normalized Gain | youtubevos_val | Three-benchmark coarse mean J-M | DAVIS17/SegTrackv2/FBMS-59 | 5 multi-view only | 0.600 | 0.802 |
| Mean Normalized Gain | youtubevos_val | Three-benchmark coarse mean J-R | DAVIS17/SegTrackv2/FBMS-59 | all 6 | 0.771 | 0.891 |
| Mean Normalized Gain | youtubevos_val | Three-benchmark coarse mean J-R | DAVIS17/SegTrackv2/FBMS-59 | 5 multi-view only | 0.600 | 0.752 |
| Mean Normalized Gain | youtubevos_val | Coarse J-M | youtubevos_val | all 6 | 1.000 | 0.974 |
| Mean Normalized Gain | youtubevos_val | Coarse J-M | youtubevos_val | 5 multi-view only | 1.000 | 0.981 |
| Mean Normalized Gain | youtubevos_val | Coarse J-R | youtubevos_val | all 6 | 0.943 | 0.978 |
| Mean Normalized Gain | youtubevos_val | Coarse J-R | youtubevos_val | 5 multi-view only | 0.900 | 0.958 |
| Coarse J-M | youtubevos_val | Three-benchmark coarse mean J-M | DAVIS17/SegTrackv2/FBMS-59 | all 6 | 0.771 | 0.957 |
| Coarse J-M | youtubevos_val | Three-benchmark coarse mean J-M | DAVIS17/SegTrackv2/FBMS-59 | 5 multi-view only | 0.600 | 0.861 |
| Coarse J-M | youtubevos_val | Three-benchmark coarse mean J-R | DAVIS17/SegTrackv2/FBMS-59 | all 6 | 0.771 | 0.930 |
| Coarse J-M | youtubevos_val | Three-benchmark coarse mean J-R | DAVIS17/SegTrackv2/FBMS-59 | 5 multi-view only | 0.600 | 0.810 |
| Coarse J-R | youtubevos_val | Three-benchmark coarse mean J-M | DAVIS17/SegTrackv2/FBMS-59 | all 6 | 0.714 | 0.969 |
| Coarse J-R | youtubevos_val | Three-benchmark coarse mean J-M | DAVIS17/SegTrackv2/FBMS-59 | 5 multi-view only | 0.500 | 0.904 |
| Coarse J-R | youtubevos_val | Three-benchmark coarse mean J-R | DAVIS17/SegTrackv2/FBMS-59 | all 6 | 0.714 | 0.943 |
| Coarse J-R | youtubevos_val | Three-benchmark coarse mean J-R | DAVIS17/SegTrackv2/FBMS-59 | 5 multi-view only | 0.500 | 0.859 |

Target coarse mean J-M is retained only to test whether a source-validation diagnostic transfers to DAVIS17, SegTrackv2, and FBMS-59.
