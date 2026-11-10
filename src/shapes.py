"""
Parametric 3D shape families with exact or converged reference surface area.

Every shape is defined in its own "body frame" with maximum extent ~1, by an
implicit function f(p) that is negative inside and positive outside. The shape
is then placed in a voxel grid with a scale s, a rotation R and a centre c.

Reference surface area
----------------------
* sphere, ellipsoid, cylinder, torus: closed form (exact).
* blob, rough_blob, superellipsoid: these are star-shaped, i.e. the surface is
  r(u) * u for unit directions u. We triangulate the surface by pushing the
  vertices of a finely subdivided icosphere out to radius r(u). The triangle
  mesh area converges as O(h^2) in the edge length h, so we compute it at two
  subdivision levels and apply Richardson extrapolation. The gap between the
  two levels is stored as a per-shape estimate of the reference error.

This avoids the trap of using marching cubes on a binary high-resolution
voxelization as "ground truth": marching cubes on binary data is not
multigrid convergent, so its bias does not vanish with resolution.
"""
from __future__ import annotations

import numpy as np
from scipy.special import ellipeinc, ellipkinc
from scipy.spatial.transform import Rotation

TRAIN_FAMILIES = ("sphere", "ellipsoid", "cylinder", "torus", "blob")
HELDOUT_FAMILIES = ("superellipsoid", "rough_blob")
RADIAL_FAMILIES = ("sphere", "ellipsoid", "blob", "rough_blob", "superellipsoid")


# --------------------------------------------------------------------------
# Icosphere and mesh utilities
# --------------------------------------------------------------------------
_ICO_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def icosphere(level: int) -> tuple[np.ndarray, np.ndarray]:
    """Unit icosphere with 20 * 4**level triangles (vectorised midpoint subdivision)."""
    if level in _ICO_CACHE:
        return _ICO_CACHE[level]
    t = (1.0 + 5 ** 0.5) / 2.0
    v = np.array([(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0),
                  (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
                  (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)], dtype=float)
    f = np.array([(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
                  (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
                  (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
                  (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)], dtype=np.int64)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    for _ in range(level):
        a, b, c = f[:, 0], f[:, 1], f[:, 2]
        edges = np.concatenate([np.stack([a, b], 1), np.stack([b, c], 1), np.stack([c, a], 1)])
        edges.sort(axis=1)
        uniq, inv = np.unique(edges, axis=0, return_inverse=True)
        inv = inv.ravel()
        mid = v[uniq[:, 0]] + v[uniq[:, 1]]
        mid /= np.linalg.norm(mid, axis=1, keepdims=True)
        n0, m = len(v), len(f)
        ab, bc, ca = n0 + inv[:m], n0 + inv[m:2 * m], n0 + inv[2 * m:]
        v = np.vstack([v, mid])
        f = np.concatenate([np.stack([a, ab, ca], 1), np.stack([b, bc, ab], 1),
                            np.stack([c, ca, bc], 1), np.stack([ab, bc, ca], 1)])
    _ICO_CACHE[level] = (v, f)
    return v, f


def mesh_area(v: np.ndarray, f: np.ndarray) -> float:
    e1 = v[f[:, 1]] - v[f[:, 0]]
    e2 = v[f[:, 2]] - v[f[:, 0]]
    return 0.5 * float(np.linalg.norm(np.cross(e1, e2), axis=1).sum())


def mesh_volume(v: np.ndarray, f: np.ndarray) -> float:
    """Enclosed volume of a closed, consistently oriented mesh (divergence theorem)."""
    return abs(float(np.einsum("ij,ij->i", v[f[:, 0]], np.cross(v[f[:, 1]], v[f[:, 2]])).sum()) / 6.0)


# --------------------------------------------------------------------------
# Closed-form areas
# --------------------------------------------------------------------------
def ellipsoid_area(a: float, b: float, c: float) -> float:
    """Exact ellipsoid surface area via incomplete elliptic integrals (Legendre form)."""
    a, b, c = sorted((a, b, c), reverse=True)
    if np.isclose(a, c):
        return 4 * np.pi * a * a
    phi = np.arccos(c / a)
    m = (a * a * (b * b - c * c)) / (b * b * (a * a - c * c))
    s = np.sin(phi)
    return float(2 * np.pi * c * c + 2 * np.pi * a * b / s *
                 (ellipeinc(phi, m) * s * s + ellipkinc(phi, m) * np.cos(phi) ** 2))


# --------------------------------------------------------------------------
# Shape definition
# --------------------------------------------------------------------------
class Shape:
    """A shape in its body frame. Subclasses implement `implicit(p)` (p: (n,3))."""

    family = "base"

    def implicit(self, p: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def area(self) -> tuple[float, float]:
        """(reference area, estimated absolute error of the reference) in body units."""
        raise NotImplementedError

    def volume(self) -> float:
        raise NotImplementedError

    def params(self) -> dict:
        return {}


class RadialShape(Shape):
    """Star-shaped surface p = r(u) u."""

    def radius(self, u: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def implicit(self, p):
        n = np.linalg.norm(p, axis=1)
        u = p / np.maximum(n, 1e-12)[:, None]
        return n - self.radius(u)

    def surface_mesh(self, level: int = 7):
        v, f = icosphere(level)
        return v * self.radius(v)[:, None], f

    def _mesh_area(self) -> tuple[float, float]:
        a6 = mesh_area(*self.surface_mesh(6))
        a7 = mesh_area(*self.surface_mesh(7))
        a_ref = (4 * a7 - a6) / 3.0          # Richardson extrapolation for O(h^2)
        return a_ref, abs(a_ref - a7)

    def area(self):
        return self._mesh_area()

    def volume(self):
        return mesh_volume(*self.surface_mesh(7))

    def normalise(self):
        """Rescale so that the maximum radius is 1 (checked on a fine icosphere)."""
        v, _ = icosphere(7)
        self.r_scale = 1.0 / float(self.radius(v).max())
        return self


class Sphere(RadialShape):
    family = "sphere"

    def radius(self, u):
        return np.ones(len(u))

    def area(self):
        return 4 * np.pi, 0.0

    def volume(self):
        return 4 / 3 * np.pi


class Ellipsoid(RadialShape):
    family = "ellipsoid"

    def __init__(self, b: float, c: float):
        self.ax = np.array([1.0, b, c])

    def radius(self, u):
        return 1.0 / np.sqrt(((u / self.ax) ** 2).sum(1))

    def area(self):
        return ellipsoid_area(*self.ax), 0.0

    def volume(self):
        return 4 / 3 * np.pi * float(np.prod(self.ax))

    def params(self):
        return {"a": 1.0, "b": float(self.ax[1]), "c": float(self.ax[2])}


class SuperEllipsoid(RadialShape):
    """|x/a|^p + |y/b|^p + |z/c|^p = 1. p=2 is an ellipsoid; p>2 boxy, p<2 pointy."""

    family = "superellipsoid"

    def __init__(self, b: float, c: float, p: float):
        self.ax = np.array([1.0, b, c])
        self.p = p
        self.r_scale = 1.0

    def radius(self, u):
        return self.r_scale * (np.abs(u / self.ax) ** self.p).sum(1) ** (-1.0 / self.p)

    def params(self):
        return {"a": 1.0, "b": float(self.ax[1]), "c": float(self.ax[2]), "p": self.p}


class Blob(RadialShape):
    """Random star-shaped blob: log r(u) is a random sum of plane waves on the sphere.

    log r(u) = sum_k amp_k * cos(freq_k * (d_k . u) + phase_k)

    With random directions d_k and phases this is an approximately Gaussian
    random field on the sphere; low frequencies give smooth blobs, adding
    high-frequency, low-amplitude waves gives a rough surface.
    """

    family = "blob"

    def __init__(self, rng: np.random.Generator, rough: bool = False):
        k_s = 6
        d = rng.normal(size=(k_s, 3))
        freq = rng.uniform(1.0, 3.0, k_s)
        amp = rng.uniform(0.5, 1.0, k_s)
        amp *= rng.uniform(0.15, 0.30) / np.sqrt((amp ** 2).sum() / 2)   # target std of log r
        phase = rng.uniform(0, 2 * np.pi, k_s)
        if rough:
            self.family = "rough_blob"
            k_r = 10
            d = np.vstack([d, rng.normal(size=(k_r, 3))])
            freq = np.concatenate([freq, rng.uniform(8.0, 14.0, k_r)])
            amp_r = rng.uniform(0.5, 1.0, k_r)
            amp_r *= rng.uniform(0.02, 0.05) / np.sqrt((amp_r ** 2).sum() / 2)
            amp = np.concatenate([amp, amp_r])
            phase = np.concatenate([phase, rng.uniform(0, 2 * np.pi, k_r)])
        self.d = d / np.linalg.norm(d, axis=1, keepdims=True)
        self.freq, self.amp, self.phase = freq, amp, phase
        self.r_scale = 1.0

    def radius(self, u):
        g = np.cos((u @ self.d.T) * self.freq + self.phase) @ self.amp
        return self.r_scale * np.exp(g)

    def params(self):
        return {"n_waves": int(len(self.freq)), "max_freq": float(self.freq.max())}


class Cylinder(Shape):
    family = "cylinder"

    def __init__(self, rho: float):
        self.rho = rho
        self.hh = np.sqrt(1.0 - rho * rho)     # max extent sqrt(rho^2 + hh^2) = 1

    def implicit(self, p):
        return np.maximum(np.hypot(p[:, 0], p[:, 1]) - self.rho, np.abs(p[:, 2]) - self.hh)

    def area(self):
        return 2 * np.pi * self.rho * (self.rho + 2 * self.hh), 0.0

    def volume(self):
        return np.pi * self.rho ** 2 * 2 * self.hh

    def params(self):
        return {"radius": self.rho, "half_height": float(self.hh)}


class Torus(Shape):
    family = "torus"

    def __init__(self, ratio: float):
        self.r = ratio / (1 + ratio)           # minor radius, R + r = 1
        self.R = 1 - self.r

    def implicit(self, p):
        return np.hypot(np.hypot(p[:, 0], p[:, 1]) - self.R, p[:, 2]) - self.r

    def area(self):
        return 4 * np.pi ** 2 * self.R * self.r, 0.0

    def volume(self):
        return 2 * np.pi ** 2 * self.R * self.r ** 2

    def params(self):
        return {"R": self.R, "r": self.r}


def make_shape(family: str, rng: np.random.Generator) -> Shape:
    if family == "sphere":
        return Sphere()
    if family == "ellipsoid":
        b, c = np.sort(rng.uniform(0.35, 1.0, 2))[::-1]
        return Ellipsoid(b, c)
    if family == "cylinder":
        return Cylinder(rng.uniform(0.3, 0.85))
    if family == "torus":
        return Torus(rng.uniform(0.2, 0.6))
    if family == "blob":
        return Blob(rng).normalise()
    if family == "rough_blob":
        return Blob(rng, rough=True).normalise()
    if family == "superellipsoid":
        b, c = np.sort(rng.uniform(0.45, 1.0, 2))[::-1]
        p = rng.uniform(1.5, 1.8) if rng.random() < 0.4 else rng.uniform(3.0, 6.0)
        return SuperEllipsoid(b, c, p).normalise()
    raise ValueError(family)


class PlacedShape:
    """A body-frame shape placed in a voxel grid of side N (units: voxels)."""

    def __init__(self, shape: Shape, N: int, fill: float, rotation: np.ndarray,
                 offset: np.ndarray, margin: float = 2.0):
        self.shape, self.N = shape, N
        self.scale = fill * (N / 2.0 - margin)
        self.R = rotation                   # body -> world
        self.centre = N / 2.0 + offset

    def implicit_world(self, x: np.ndarray) -> np.ndarray:
        """Signed implicit value in voxel units (negative inside)."""
        p = ((x - self.centre) @ self.R) / self.scale    # world -> body: R^T (x - c)
        return self.shape.implicit(p) * self.scale

    def area(self) -> tuple[float, float]:
        a, err = self.shape.area()
        return a * self.scale ** 2, err * self.scale ** 2

    def volume(self) -> float:
        return self.shape.volume() * self.scale ** 3


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    return Rotation.random(random_state=int(rng.integers(2 ** 31))).as_matrix()
