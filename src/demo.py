"""
Demo: estimate the surface area of a voxel volume with every method.

  python src/demo.py --volume my_volume.npy        # your own volume
  python src/demo.py --example blob --N 32 --noise 0.15  # synthetic example with known area

A user volume should be a cubic float array with background ~0 and object ~1
(or uint8 0..255, which is rescaled to 0..1). The learned models are applied
only to 32^3 or 64^3 grids, the sizes they were trained on.
"""
from __future__ import annotations

import argparse

import numpy as np

from infer import estimate_all
from shapes import PlacedShape, make_shape, random_rotation
from voxelize import ct_image, decode, encode, occupancy

NAMES = {"face": "Voxel-face counting", "mc_bin": "Marching cubes (binary)",
         "mc_grey": "Marching cubes (greyscale)", "mc_smooth": "Marching cubes (smoothed)",
         "cnn": "3D CNN correction", "pointnet": "PointNet correction"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--volume", help=".npy cubic volume")
    ap.add_argument("--example", choices=["sphere", "ellipsoid", "cylinder", "torus", "blob",
                                          "superellipsoid", "rough_blob"])
    ap.add_argument("--N", type=int, default=32)
    ap.add_argument("--noise", type=float, default=0.05)
    ap.add_argument("--voxel_size", type=float, default=1.0, help="physical edge length of one voxel")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--root", default=".")
    args = ap.parse_args()

    a_true = None
    if args.volume:
        img = np.load(args.volume)
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        img = img.astype(np.float32)
        assert img.ndim == 3 and len(set(img.shape)) == 1, "volume must be cubic (N, N, N)"
    elif args.example:
        rng = np.random.default_rng(args.seed)
        ps = PlacedShape(make_shape(args.example, rng), args.N, rng.uniform(0.35, 1.0),
                         random_rotation(rng), rng.uniform(-0.5, 0.5, 3))
        img = decode(encode(ct_image(occupancy(ps), args.noise, rng)))
        a_true = ps.area()[0]
        print(f"synthetic {args.example}, {args.N}^3 grid, noise {args.noise}, "
              f"diameter {2 * ps.scale:.1f} voxels")
    else:
        ap.error("give --volume or --example")

    res = estimate_all(img, args.root)
    s = args.voxel_size ** 2
    print(f"\n{'method':30s} {'area':>12s}" + (f" {'error':>8s}" if a_true else ""))
    if a_true:
        print(f"{'reference (exact)':30s} {a_true * s:12.1f}")
    for k, a in res.items():
        line = f"{NAMES[k]:30s} {a * s:12.1f}"
        if a_true:
            line += f" {100 * (a / a_true - 1):+7.2f}%"
        print(line)
    if "cnn" not in res:
        print(f"\n(no learned model for a {img.shape[0]}^3 grid; trained sizes are 32 and 64)")


if __name__ == "__main__":
    main()
