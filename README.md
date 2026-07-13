# SalientGS: Unified SfM-to-3DGS with Importance-Guided MCMC Gaussian Allocation

SalientGS is an end-to-end pipeline for reconstructing 3D Gaussian Splatting (3DGS) scenes from unordered images. It combines Fisher Vector retrieval with MST connectivity, first-order SfM, joint pose-appearance refinement, and importance-guided MCMC Gaussian allocation.

Project page: <https://six-bit-tx.github.io/SalientGS/>

Paper: [`paper/main.pdf`](paper/main.pdf) · Supplementary: [`paper/supplementary.pdf`](paper/supplementary.pdf)

## Highlights

- Importance-guided MCMC reallocates a fixed Gaussian budget toward persistent multi-view underfit regions.
- The unified pipeline jointly refines SfM poses and Gaussian appearance with photometric and reprojection losses.
- Fisher Vector retrieval plus MST connectivity provides a fast unordered-image front end.
- The released implementation includes the FastMap CUDA extension, evaluation scripts, and command-line entry points.

## Installation

The released code targets Python 3.10 or newer, PyTorch with CUDA support, COLMAP for SIFT feature extraction, and [`gsplat`](https://github.com/nerfstudio-project/gsplat).

```bash
git clone https://github.com/Six-Bit-TX/SalientGS.git
cd SalientGS

pip install -e fastmap/
pip install -e .
```

The Python package dependencies are pinned or listed in [`pyproject.toml`](pyproject.toml). The FastMap extension requires a CUDA-capable build environment.

## Usage

```bash
DATA=/path/to/dataset

# Feature extraction and Fisher Vector retrieval
sgs-feat --image_dir "$DATA/images" --output_dir "$DATA"

# First-order SfM
sgs-sfm --headless \
  --database "$DATA/database.db" \
  --image_dir "$DATA/images" \
  --output_dir "$DATA"

# Joint 3DGS training
sgs-joint --data_path "$DATA"
```

The expected dataset layout is:

```text
dataset/
├── images/
├── database.db
└── sparse/0/
    ├── cameras.txt
    ├── images.txt
    └── points3D.txt
```

The [`scripts/`](scripts/) directory contains benchmark, runtime, ETH3D pose-evaluation, and GLOMAP comparison scripts.

## Reproducibility notes

The paper reports a 1.5M Gaussian budget, 30K joint-training iterations, Fisher Vector retrieval with 64 GMM components and top-20 neighbors, robust score quantiles `(0.05, 0.90)`, an importance threshold of 5, redundancy threshold of 0.9, opacity mixing of 0.05, and 10 views per score update. Scores begin after a 3K pose warmup and are recomputed every 500 iterations.

Datasets, pretrained weights, and generated experiment outputs are not included in this repository. Please follow the licenses and terms of the respective datasets and third-party dependencies.

## License

The code in this repository is released under [CC BY-NC 4.0](LICENSE). Third-party components retain their own licenses.

## Citation

```bibtex
@inproceedings{xiong2026salientgs,
  author    = {Tianyu Xiong and Rui Li and Suning Ge and Jiaqi Yang},
  title     = {SalientGS: Unified SfM-to-3DGS with Importance-Guided MCMC Gaussian Allocation},
  booktitle = {Proceedings of the 34th ACM International Conference on Multimedia},
  year      = {2026}
}
```
