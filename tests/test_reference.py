"""
Checks of the reference-area and voxelization pipeline.

  python tests/test_reference.py      -> prints PASS/FAIL, writes results/reference_checks.csv
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd

from classical import all_estimates
from shapes import (RadialShape, PlacedShape, Sphere, SuperEllipsoid, make_shape, mesh_area,
                    random_rotation)
from voxelize import ct_image, decode, encode, occupancy

rows = []


def check(name, value, limit, note=""):
    ok = abs(value) <= limit
    rows.append({"check": name, "value": value, "limit": limit, "pass": ok, "note": note})
    print(f"{'PASS' if ok else 'FAIL'}  {name:58s} {value: .3e}  (limit {limit:g}) {note}")


rng = np.random.default_rng(0)

# 1. Mesh-based reference vs exact closed form
a, _ = RadialShape._mesh_area(Sphere())
check("sphere: mesh reference vs 4*pi*r^2 (relative)", a / (4 * np.pi) - 1, 1e-6)
worst = 0.0
for _ in range(20):
    e = make_shape("ellipsoid", rng)
    worst = max(worst, abs(RadialShape._mesh_area(e)[0] / e.area()[0] - 1))
check("ellipsoid (20 random): mesh reference vs elliptic integrals", worst, 1e-6)

# 2. Convergence of the Richardson reference for the hardest shapes
for fam, shp in [("superellipsoid p=1.5", SuperEllipsoid(0.7, 0.5, 1.5).normalise()),
                 ("superellipsoid p=6", SuperEllipsoid(0.7, 0.5, 6.0).normalise()),
                 ("rough blob", make_shape("rough_blob", rng))]:
    a6, a7, a8 = (mesh_area(*shp.surface_mesh(k)) for k in (6, 7, 8))
    check(f"{fam}: Richardson(6,7) vs Richardson(7,8)", ((4 * a7 - a6) / 3) / ((4 * a8 - a7) / 3) - 1, 1e-4)

# 3. Partial-volume rasterizer conserves volume
for fam in ("sphere", "ellipsoid", "cylinder", "torus", "blob", "superellipsoid", "rough_blob"):
    ps = PlacedShape(make_shape(fam, rng), 64, 0.8, random_rotation(rng), rng.uniform(-0.5, 0.5, 3))
    check(f"{fam}: rasterized volume vs reference volume (64^3)", occupancy(ps).sum() / ps.volume() - 1, 1e-3)

# 4. Known behaviour of classical estimators on a clean sphere
ps = PlacedShape(Sphere(), 128, 0.9, random_rotation(rng), rng.uniform(-0.5, 0.5, 3))
est = all_estimates(decode(encode(ct_image(occupancy(ps), 0.0, rng))))
A = ps.area()[0]
check("sphere 128^3: face counting error minus theoretical +50%", est["A_face"] / A - 1.5, 0.02)
check("sphere 128^3: greyscale marching cubes error", est["A_mc_grey"] / A - 1, 0.005)

os.makedirs("results", exist_ok=True)
pd.DataFrame(rows).to_csv("results/reference_checks.csv", index=False)
print(f"\n{sum(r['pass'] for r in rows)}/{len(rows)} checks passed")
