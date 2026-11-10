"""
Bias-vs-resolution sweep (Experiment 3).

The same test-split shapes (same body shape, rotation and fill fraction) are
rendered at N = 16 ... 128 at three noise levels. Classical estimators run at
every N; the learned models run at the N they were trained for (32, 64).

  python src/sweep.py --per_family 8 --workers 6
"""
from __future__ import annotations

import argparse
import os
from multiprocessing import Pool

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from build_dataset import FILL_RANGE, SEED, plan_shapes
from evaluate import COLOR, LABEL
from shapes import PlacedShape, make_shape, random_rotation
from voxelize import NOISE_LEVELS, ct_image, decode, encode, occupancy

SWEEP_N = (16, 24, 32, 48, 64, 96, 128)
ROOT = "."


def run_shape(job):
    try:
        from infer import estimate_all        # imported in the worker (loads torch)
    except ImportError:                       # no torch: classical estimators only
        from classical import all_estimates

        def estimate_all(img, root):
            e = all_estimates(img)
            return {"face": e["A_face"], "mc_bin": e["A_mc_bin"], "mc_grey": e["A_mc_grey"],
                    "mc_smooth": e["A_mc_smooth"]}
    rng = np.random.default_rng([SEED, job["shape_id"]])
    shape = make_shape(job["family"], rng)    # identical draw order to build_dataset
    fill = rng.uniform(*FILL_RANGE)
    rot = random_rotation(rng)
    rows = []
    for N in SWEEP_N:
        ps = PlacedShape(shape, N, fill, rot, np.zeros(3))
        a_true, _ = ps.area()
        occ = occupancy(ps)
        for j, sn in enumerate(NOISE_LEVELS):
            img = decode(encode(ct_image(occ, sn, np.random.default_rng([SEED, job["shape_id"], N, j, 7]))))
            for m, a in estimate_all(img, ROOT).items():
                rows.append({"shape_id": job["shape_id"], "family": job["family"], "N": N, "noise": sn,
                             "diam_vox": 2 * ps.scale, "method": m, "err": a / a_true - 1})
    return rows


def plot(df, root):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    classical = ["mc_bin", "mc_smooth", "mc_grey"]
    for ax, sn in zip(axes, NOISE_LEVELS):
        d = df[df.noise == sn]
        for m in classical:
            g = d[d.method == m].groupby("N").err
            mu, sd = g.mean() * 100, g.std() * 100
            ax.plot(mu.index, mu.values, "-o", color=COLOR[m], lw=2, ms=4, label=LABEL[m])
            ax.fill_between(mu.index, mu - sd, mu + sd, color=COLOR[m], alpha=0.12, lw=0)
        for k, m in enumerate(("cnn", "pointnet")):
            g = d[d.method == m].groupby("N").err
            if len(g):
                mu, sd = g.mean() * 100, g.std() * 100
                ax.errorbar(mu.index * (1.04 + 0.04 * k), mu.values, yerr=sd.values, fmt="s", color=COLOR[m],
                            ms=6, capsize=3, lw=1.5, label=LABEL[m])
        ax.axhline(0, color="#52514e", lw=0.8)
        ax.set_xscale("log", base=2)
        ax.set_xticks(SWEEP_N, [str(n) for n in SWEEP_N])
        ax.set_title(f"noise σ = {sn:g}", fontsize=10)
    axes[0].set_ylabel("mean signed error (%)  ± 1 SD")
    for ax in axes:
        ax.set_xlabel("grid size N")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=6, fontsize=8)
    fig.suptitle("Bias vs resolution (voxel-face counting stays at about +50% at every N, not shown)", fontsize=10, y=1.03)
    fig.savefig(os.path.join(root, "figures", "bias_vs_resolution.png"))
    plt.close(fig)


def main():
    global ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--per_family", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--plot_only", action="store_true", help="redraw the figure from results/sweep.csv")
    args = ap.parse_args()
    ROOT = args.root
    if args.plot_only:
        plot(pd.read_csv(os.path.join(args.root, "results", "sweep.csv")), args.root)
        return
    os.makedirs(os.path.join(args.root, "figures"), exist_ok=True)
    os.makedirs(os.path.join(args.root, "results"), exist_ok=True)
    jobs, count = [], {}
    for j in plan_shapes():
        if j["split"] in ("test", "held_super") and count.get(j["family"], 0) < args.per_family:
            jobs.append(j)
            count[j["family"]] = count.get(j["family"], 0) + 1
    print(f"sweep over {len(jobs)} shapes x {len(SWEEP_N)} resolutions x {len(NOISE_LEVELS)} noise levels", flush=True)
    rows = []
    with Pool(args.workers) as pool:
        for k, r in enumerate(pool.imap_unordered(run_shape, jobs)):
            rows += r
            print(f"{k + 1}/{len(jobs)}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.root, "results", "sweep.csv"), index=False)
    summ = df.groupby(["noise", "method", "N"]).err.agg(bias=lambda e: 100 * e.mean(),
                                                       mape=lambda e: 100 * e.abs().mean()).reset_index()
    summ.to_csv(os.path.join(args.root, "results", "sweep_summary.csv"), index=False)
    print(summ[summ.noise == 0].pivot(index="N", columns="method", values="bias").round(2).to_string())
    plot(df, args.root)


if __name__ == "__main__":
    main()
