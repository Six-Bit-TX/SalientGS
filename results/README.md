# Paper-protocol verification

The archived CSV records an end-to-end run of all 13 paper scenes with seed 42,
30,000 optimization steps, a 1.5M Gaussian cap, top-20 retrieval, and a 64-component
Fisher Vector GMM. Feature extraction, matching, SfM, and joint SalientGS training
were all run by `scripts/reproduce_paper.sh`; reference poses were not used.

## Aggregate results

The table reports simple scene means. Runtime includes feature extraction, SfM,
and training on one NVIDIA RTX PRO 6000 Blackwell GPU.

| Dataset | Scenes | PSNR | SSIM | LPIPS | Mean runtime |
|---|---:|---:|---:|---:|---:|
| Mip-NeRF 360 | 9 | 28.822 | 0.853 | 0.148 | 11.79 min |
| Deep Blending | 2 | 29.488 | 0.906 | 0.183 | 10.04 min |
| Tanks and Temples | 2 | 24.655 | 0.869 | 0.109 | 10.03 min |

For Mip-NeRF 360, pooling by held-out image count gives 29.607 PSNR, 0.874
SSIM, and 0.140 LPIPS. Different papers may use either scene means or pooled
image means, so both are stated explicitly rather than mixed.

The run is a reproducibility record, not a replacement for the camera-ready
paper table: hardware, library versions, scene aggregation, and upstream metric
implementations can shift the final decimals. Raw scene values are retained in
the CSV so comparisons can be recomputed without relying on rounded aggregates.
