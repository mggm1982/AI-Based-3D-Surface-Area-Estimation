"""
Classical (non-learned) surface area estimators and hand-crafted descriptors.

Estimators, all returning area in voxel^2:
  face_count   : threshold at 0.5, count exposed voxel faces (staircase area).
  mc_binary    : marching cubes on the thresholded (binary) volume.
  mc_grey      : marching cubes on the greyscale volume at iso-level 0.5,
                 which uses sub-voxel interpolation of the partial-volume edge.
  mc_smooth    : Gaussian pre-smoothing (sigma = 1 voxel) then marching cubes
                 at 0.5; a standard denoising step before mesh extraction.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.measure import euler_number, marching_cubes, mesh_surface_area

ISO = 0.5


def face_count_area(binary: np.ndarray) -> float:
    b = np.pad(binary.astype(np.int8), 1)
    return float(sum(np.abs(np.diff(b, axis=ax)).sum() for ax in range(3)))


def mc_mesh(vol: np.ndarray, level: float = ISO):
    v = np.pad(vol.astype(np.float32), 1, constant_values=0.0)
    if v.max() <= level or v.min() >= level:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int64)
    verts, faces, _, _ = marching_cubes(v, level=level)
    return verts - 1.0, faces          # undo padding offset


def mc_area(vol: np.ndarray, level: float = ISO) -> float:
    verts, faces = mc_mesh(vol, level)
    return float(mesh_surface_area(verts, faces)) if len(faces) else 0.0


def noise_estimate(img: np.ndarray) -> float:
    """Background noise std: voxels far from the object (smoothed value < 0.05)."""
    bg = gaussian_filter(img, 2.0) < 0.05
    return float(img[bg].std()) if bg.sum() > 100 else 0.0


def shape_descriptors(img: np.ndarray, binary: np.ndarray) -> dict:
    """Volume, principal-axis aspect ratios and Euler number."""
    w = np.clip(img, 0, 1)
    vol = float(w.sum())
    idx = np.argwhere(binary)
    if len(idx) > 10:
        ev = np.sort(np.linalg.eigvalsh(np.cov(idx.T)))[::-1]
        ev = np.maximum(ev, 1e-6)
        ar1, ar2 = float(np.sqrt(ev[1] / ev[0])), float(np.sqrt(ev[2] / ev[0]))
    else:
        ar1 = ar2 = 1.0
    return {"vol_occ": vol, "vol_bin": float(binary.sum()), "aspect1": ar1, "aspect2": ar2,
            "euler": float(euler_number(binary, connectivity=3))}


def all_estimates(img: np.ndarray) -> dict:
    binary = img > ISO
    out = {
        "A_face": face_count_area(binary),
        "A_mc_bin": mc_area(binary.astype(np.float32)),
        "A_mc_grey": mc_area(img),
        "A_mc_smooth": mc_area(gaussian_filter(img, 1.0)),
        "noise_hat": noise_estimate(img),
    }
    out.update(shape_descriptors(img, binary))
    return out


def sample_surface_points(verts: np.ndarray, faces: np.ndarray, n: int,
                          rng: np.random.Generator) -> np.ndarray:
    """Area-weighted uniform sampling of n points on a triangle mesh."""
    if len(faces) == 0:
        return np.zeros((n, 3), np.float32)
    tri = verts[faces]
    w = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    k = rng.choice(len(faces), size=n, p=w / w.sum())
    r1, r2 = rng.random(n), rng.random(n)
    s = np.sqrt(r1)
    a, b, c = 1 - s, s * (1 - r2), s * r2
    t = tri[k]
    return (a[:, None] * t[:, 0] + b[:, None] * t[:, 1] + c[:, None] * t[:, 2]).astype(np.float32)
