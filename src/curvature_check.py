"""
Clean spheres: measured classical errors vs the curvature prediction -2 sigma_eff^2 / R^2.

  python src/curvature_check.py   -> results/sphere_curvature.csv (Table 1 of the article)
"""
import os

import numpy as np
import pandas as pd

from classical import all_estimates
from shapes import PlacedShape, Sphere, random_rotation
from voxelize import PSF_SIGMA, ct_image, decode, encode, occupancy

S2_GREY = PSF_SIGMA ** 2 + 1 / 12          # Gaussian PSF + voxel box
S2_SMOOTH = S2_GREY + 1.0                  # + smoothing filter, sigma_s = 1

rng = np.random.default_rng(1)
rows = []
for N in (16, 32, 64, 128):
    ps = PlacedShape(Sphere(), N, 0.9, random_rotation(rng), rng.uniform(-0.5, 0.5, 3))
    occ = occupancy(ps)
    e = all_estimates(decode(encode(ct_image(occ, 0.0, rng))))
    A, R = ps.area()[0], ps.scale
    rows.append({"N": N, "R_vox": R,
                 "face_%": 100 * (e["A_face"] / A - 1), "mc_bin_%": 100 * (e["A_mc_bin"] / A - 1),
                 "mc_grey_%": 100 * (e["A_mc_grey"] / A - 1), "mc_grey_pred_%": -200 * S2_GREY / R ** 2,
                 "mc_smooth_%": 100 * (e["A_mc_smooth"] / A - 1), "mc_smooth_pred_%": -200 * S2_SMOOTH / R ** 2})
df = pd.DataFrame(rows)
os.makedirs("results", exist_ok=True)
df.to_csv("results/sphere_curvature.csv", index=False)
print(df.round(2).to_string(index=False))
