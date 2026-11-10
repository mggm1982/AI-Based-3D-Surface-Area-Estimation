"""
CT-like voxelization of a placed shape.

Forward model (all in voxel units):
  1. Partial-volume occupancy: each voxel value is the fraction of the voxel
     inside the shape, computed by 5x5x5 supersampling. Only voxels in a band
     around the surface are supersampled; the rest are exactly 0 or 1.
  2. System blur: Gaussian point-spread function, sigma = 0.6 voxel.
  3. Additive Gaussian noise with standard deviation sigma_n (relative to a
     unit attenuation contrast between object and background).

Volumes are stored as uint8 with the linear map  q = round((I + 0.5) * 127.5),
which covers intensities in [-0.5, 1.5] with a step of 0.0078.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from shapes import PlacedShape

PSF_SIGMA = 0.6
NOISE_LEVELS = (0.0, 0.05, 0.15)
SUPERSAMPLE = 5
BAND = 4.0           # voxels; supersample where |f(centre)| < BAND


def voxel_centres(N: int) -> np.ndarray:
    g = np.arange(N) + 0.5
    x, y, z = np.meshgrid(g, g, g, indexing="ij")
    return np.stack([x.ravel(), y.ravel(), z.ravel()], 1)


def occupancy(ps: PlacedShape, ss: int = SUPERSAMPLE, chunk: int = 4000) -> np.ndarray:
    N = ps.N
    c = voxel_centres(N)
    f = ps.implicit_world(c)
    occ = (f < 0).astype(np.float32)
    band = np.flatnonzero(np.abs(f) < BAND)
    o = (np.arange(ss) + 0.5) / ss - 0.5
    ox, oy, oz = np.meshgrid(o, o, o, indexing="ij")
    offs = np.stack([ox.ravel(), oy.ravel(), oz.ravel()], 1)          # (ss^3, 3)
    for i in range(0, len(band), chunk):
        idx = band[i:i + chunk]
        pts = (c[idx, None, :] + offs[None]).reshape(-1, 3)
        inside = ps.implicit_world(pts).reshape(len(idx), -1) < 0
        occ[idx] = inside.mean(1)
    return occ.reshape(N, N, N)


def ct_image(occ: np.ndarray, sigma_n: float, rng: np.random.Generator,
             psf: float = PSF_SIGMA) -> np.ndarray:
    img = gaussian_filter(occ.astype(np.float32), psf, mode="constant")
    if sigma_n > 0:
        img = img + rng.normal(0.0, sigma_n, img.shape).astype(np.float32)
    return img


def encode(img: np.ndarray) -> np.ndarray:
    return np.clip(np.rint((img + 0.5) * 127.5), 0, 255).astype(np.uint8)


def decode(q: np.ndarray) -> np.ndarray:
    return q.astype(np.float32) / 127.5 - 0.5
