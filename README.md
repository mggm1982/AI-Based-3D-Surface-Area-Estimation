# Learned bias correction for surface-area estimation from CT-like voxel data

Surface area measured from voxelized CT data is biased. Counting exposed voxel
faces overestimates the area by about 50% at every resolution, and marching
cubes is biased by thresholding, noise and the finite voxel size. This project
builds synthetic CT-like volumes of shapes with exactly known surface area and
trains a 3D CNN and a PointNet to predict a multiplicative correction to the
greyscale marching-cubes area. They
are benchmarked against four classical estimators and two calibrated baselines
(a per-regime lookup factor and a Random Forest).

## Main result (test shapes never seen in training, mean absolute error)

| Method | 32³ | 64³ |
|---|---|---|
| Voxel-face counting | 56.71% | 65.54% |
| Marching cubes, binary | 10.38% | 14.49% |
| Marching cubes, greyscale | 4.78% | 3.52% |
| MC + lookup factor | 2.97% | 0.79% |
| PointNet | 1.98% | 1.04% |
| Random Forest | 0.72% | **0.18%** |
| 3D CNN | **0.68%** | 0.26% |

Full tables: `results/tables.md`; article: `paper/Project51_ManasJain_Article.pdf`.

## Repository layout

```
src/
  shapes.py         shape families and reference (exact or converged) surface area
  voxelize.py       partial-volume rasterizer, Gaussian PSF, Gaussian noise, uint8 encoding
  classical.py      voxel-face counting, marching cubes (binary / greyscale / smoothed), descriptors
  build_dataset.py  generates training_data/ and testing_data/
  data.py           loading, targets, augmentation
  models.py         CNN3D and PointNet
  train.py          training one model at one resolution
  baselines.py      MC + lookup table, Random Forest on hand-crafted descriptors
  evaluate.py       metrics, tables and figures for all estimators
  sweep.py          bias-vs-resolution sweep, N = 16 ... 128
  infer.py, demo.py estimate the area of a single volume
  figures_extra.py  explanatory figures (example slices, meshes, face-count theory)
  curvature_check.py  curvature-bias prediction vs measurement on spheres
  report_tables.py  LaTeX tables for the article, generated from results/
tests/
  test_reference.py checks of the reference-area and voxelization pipeline
weights/            trained model weights
results/            CSV tables of predictions and metrics, tables.md
figures/            plots used in the report
paper/              preprint article (LaTeX source and PDF)
slides/             presentation: Beamer source (beamer/slides.tex), PDF, and the .pptx made from it
```

## Setup

```bash
conda create -y -n p51 python=3.11 && conda activate p51
pip install -r requirements.txt
```

## Reproducing everything

```bash
python tests/test_reference.py                               # pipeline checks
python src/build_dataset.py --out . --workers 6              # dataset (seeded, deterministic)
python src/baselines.py                                      # lookup and Random Forest baselines
python src/train.py --model cnn --N 32
python src/train.py --model cnn --N 64
python src/train.py --model pointnet --N 32
python src/train.py --model pointnet --N 64
python src/evaluate.py                                       # tables and figures
python src/sweep.py --per_family 8 --workers 6               # resolution sweep
python src/curvature_check.py                                # sphere curvature table
python src/figures_extra.py                                  # explanatory figures
python src/report_tables.py                                  # LaTeX tables for the article
(cd slides/beamer && latexmk -pdf slides.tex)                # slides (Beamer PDF)
python slides/make_pptx.py                                   # .pptx from the Beamer PDF
```

Training used PyTorch's MPS backend on an Apple-silicon laptop (about 60 s per
epoch for the 32³ CNN, about 7 min per epoch at 64³).

Demo on a synthetic shape with known area, or on your own `.npy` volume:

```bash
python src/demo.py --example rough_blob --N 32 --noise 0.15
python src/demo.py --volume my_volume.npy --voxel_size 0.05
```

## Data

| | |
|---|---|
| Shape families (training) | sphere, ellipsoid, cylinder, torus, smooth random blob |
| Held-out families (test only) | superellipsoid, rough random blob |
| Shape instances | 2,000 training-family (70/15/15 split by shape) + 200 + 200 held-out |
| Variants per shape | grid N = 32 and 64, noise σ = 0, 0.05, 0.15 (6 volumes per shape) |
| Object size | fills 35-100% of the grid (diameter about 10 to 60 voxels) |
| Volume format | `vol_{split}_N{N}.npy`, uint8, intensity = q / 127.5 - 0.5 |
| Point clouds | `pts_{split}_N{N}.npy`, 2,048 points sampled on the greyscale marching-cubes mesh |
| Labels | `labels_{split}.csv`: reference area, classical estimates, descriptors |

Reference areas are closed form for spheres, ellipsoids (incomplete elliptic
integrals), cylinders and tori. Blobs and superellipsoids are star-shaped, so
their surface is triangulated on a level-6 and level-7 icosphere and the mesh
areas are Richardson-extrapolated; the stored `A_ref_err` is below 0.01% of the
area for every shape.
