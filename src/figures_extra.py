"""
Explanatory figures that do not depend on trained models.

  python src/figures_extra.py
    figures/example_slices.png   central slices of one shape at N = 32 / 64 and three noise levels
    figures/example_meshes.png   reference surface vs binary / greyscale marching-cubes meshes
    figures/face_count_theory.png  voxel-face bias vs plane orientation (|nx|+|ny|+|nz|)
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from classical import mc_mesh
from shapes import PlacedShape, make_shape, random_rotation
from voxelize import NOISE_LEVELS, ct_image, decode, encode, occupancy

OUT = "figures"
INK, MUTED = "#0b0b0b", "#52514e"


def example_slices():
    rng = np.random.default_rng(7)
    shape = make_shape("blob", rng)
    rot = random_rotation(rng)
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.0))
    for r, N in enumerate((32, 64)):
        ps = PlacedShape(shape, N, 0.8, rot, np.zeros(3))
        occ = occupancy(ps)
        for c, sn in enumerate(NOISE_LEVELS):
            img = decode(encode(ct_image(occ, sn, np.random.default_rng([7, N, c]))))
            ax = axes[r, c]
            ax.imshow(img[:, :, N // 2].T, cmap="gray", vmin=-0.3, vmax=1.3, origin="lower",
                      interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f"noise σ = {sn:g}", fontsize=10)
            if c == 0:
                ax.set_ylabel(f"{N}³ grid", fontsize=10)
    fig.suptitle("Central slice of the same random blob (partial volume + PSF blur + noise)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "example_slices.png"), dpi=200)
    plt.close(fig)


def _draw(ax, v, f, title, color):
    tri = v[f]
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    light = np.array([-0.5, -0.4, 0.75]) / np.linalg.norm([-0.5, -0.4, 0.75])
    shade = 0.35 + 0.65 * np.abs(nrm @ light)                # simple two-sided Lambert shading
    rgb = np.array(matplotlib.colors.to_rgb(color))
    ax.add_collection3d(Poly3DCollection(tri, facecolors=np.clip(shade[:, None] * rgb, 0, 1),
                                         edgecolors="none"))
    lo, hi = v.min(0), v.max(0)
    c, r = (lo + hi) / 2, (hi - lo).max() / 2
    ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r); ax.set_zlim(c[2] - r, c[2] + r)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    ax.set_title(title, fontsize=9)


def example_meshes():
    rng = np.random.default_rng(11)
    shape = make_shape("rough_blob", rng)
    rot = random_rotation(rng)
    N = 32
    ps = PlacedShape(shape, N, 0.85, rot, np.zeros(3))
    a_true = ps.area()[0]
    vr, fr = shape.surface_mesh(5)
    vr = (vr * ps.scale) @ rot.T + ps.centre - 0.5          # world coords, voxel-index frame
    occ = occupancy(ps)
    img = decode(encode(ct_image(occ, 0.0, rng)))
    vb, fb = mc_mesh((img > 0.5).astype(np.float32))
    vg, fg = mc_mesh(img)

    def area(v, f):
        return 0.5 * np.linalg.norm(np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]), axis=1).sum()

    fig = plt.figure(figsize=(9, 3.4))
    items = [(vr, fr, f"reference surface\nA = {a_true:.0f} voxel²", "#c3c2b7"),
             (vb, fb, f"MC on binary volume\n{100 * (area(vb, fb) / a_true - 1):+.1f}%", "#008300"),
             (vg, fg, f"MC on greyscale volume\n{100 * (area(vg, fg) / a_true - 1):+.1f}%", "#2a78d6")]
    for k, (v, f, t, col) in enumerate(items):
        _draw(fig.add_subplot(1, 3, k + 1, projection="3d"), v, f, t, col)
    fig.suptitle(f"Rough blob on a {N}³ grid, no noise: staircase vs interpolated surfaces", fontsize=10)
    fig.savefig(os.path.join(OUT, "example_meshes.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def face_count_theory():
    """|nx|+|ny|+|nz| is the face-count area per unit true area for a plane with normal n."""
    th = np.linspace(0, np.pi / 2, 181)
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    for phi_deg, ls in ((0, "-"), (45, "--")):
        phi = np.radians(phi_deg)
        n = np.stack([np.sin(th) * np.cos(phi), np.sin(th) * np.sin(phi), np.cos(th)])
        ax.plot(np.degrees(th), np.abs(n).sum(0), ls, color="#2a78d6", lw=2, label=f"azimuth {phi_deg}°")
    ax.axhline(1.5, color="#e34948", lw=1.5, label="isotropic mean = 3/2")
    ax.set_xlabel("polar angle of surface normal (°)")
    ax.set_ylabel("face-count area / true area")
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "face_count_theory.png"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    example_slices()
    example_meshes()
    face_count_theory()
    print("wrote figures/example_slices.png, example_meshes.png, face_count_theory.png")
