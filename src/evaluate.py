"""
Evaluate every estimator on every evaluation split and write tables + figures.

  python src/evaluate.py

Reads results/preds_*.csv (lookup, rf, cnn_N*, pointnet_N*) together with the
classical estimates stored in the label files. Writes
  results/long_errors.csv      one row per (sample, method)
  results/metrics_*.csv        grouped metrics
  results/tables.md            the tables used in the report
  figures/*.png
"""
from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

from data import EVAL_SPLITS, load_labels

# Fixed method order = fixed colour slot (reference categorical palette, light mode).
METHODS = ["mc_grey", "cnn", "pointnet", "rf", "mc_smooth", "mc_bin", "lookup", "face"]
LABEL = {"mc_grey": "MC (greyscale)", "cnn": "3D CNN", "pointnet": "PointNet",
         "rf": "Random Forest", "mc_smooth": "MC (smoothed)", "mc_bin": "MC (binary)",
         "lookup": "MC + lookup", "face": "Voxel faces"}
COLOR = dict(zip(METHODS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                           "#e87ba4", "#008300", "#6250d6", "#8a8986"]))
# voxel-face counting (~+50%) is reported in tables only, never plotted.
CLASSICAL_COL = {"face": "A_face", "mc_bin": "A_mc_bin", "mc_grey": "A_mc_grey", "mc_smooth": "A_mc_smooth"}
SPLIT_LABEL = {"val": "Validation", "test": "Test (seen families)",
               "held_super": "Held-out: superellipsoids", "held_rough": "Held-out: rough blobs"}

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b",
                     "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "savefig.dpi": 200, "savefig.bbox": "tight",
                     "legend.frameon": False})


def long_table(root: str) -> pd.DataFrame:
    labs = pd.concat([load_labels(root, s) for s in EVAL_SPLITS], ignore_index=True)
    key = ["split", "N", "index"]
    out = []
    for m, col in CLASSICAL_COL.items():
        d = labs.copy()
        d["method"], d["A_est"] = m, d[col]
        out.append(d)
    for m in ("lookup", "rf", "cnn", "pointnet"):
        files = [f for f in os.listdir(os.path.join(root, "results"))
                 if f == f"preds_{m}.csv" or f.startswith(f"preds_{m}_N")]
        if not files:
            print(f"(no predictions for {m}, skipped)")
            continue
        p = pd.concat([pd.read_csv(os.path.join(root, "results", f)) for f in files])
        d = labs.merge(p[key + ["A_pred"]], on=key, how="inner")
        d["method"], d["A_est"] = m, d["A_pred"]
        out.append(d.drop(columns="A_pred"))
    df = pd.concat(out, ignore_index=True)
    df["err"] = df.A_est / df.A_true - 1
    return df[["split", "N", "index", "shape_id", "family", "noise", "noise_idx", "diam_vox",
               "A_true", "method", "A_est", "err"]]


def metrics(g: pd.DataFrame) -> pd.Series:
    e = g.err.values * 100
    a, t = g.A_est.values, g.A_true.values
    r2 = 1 - np.sum((a - t) ** 2) / np.sum((t - t.mean()) ** 2)
    return pd.Series({"MAPE": np.mean(np.abs(e)), "bias": np.mean(e), "sd": np.std(e),
                      "P95": np.percentile(np.abs(e), 95), "R2": r2, "n": len(e)})


def grouped(df, by):
    return df.groupby(by).apply(metrics, include_groups=False).reset_index()


FIG_EXCLUDE = {"face"}


def methods_present(df):
    return [m for m in METHODS if m in set(df.method)]


def fmt_pivot(t: pd.DataFrame, index, value="MAPE", digits=2) -> str:
    p = t.pivot_table(index=index, columns="method", values=value)
    p = p[[m for m in METHODS if m in p.columns]].rename(columns=LABEL)
    return p.round(digits).to_markdown()


def fig_noise(df, root):
    ms = [m for m in methods_present(df) if m not in FIG_EXCLUDE]
    t = grouped(df[df.split == "test"], ["N", "noise", "method"])
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    w = 0.8 / len(ms)
    for ax, N in zip(axes, sorted(t.N.unique())):
        tt = t[t.N == N]
        for k, m in enumerate(ms):
            v = tt[tt.method == m].sort_values("noise")
            ax.bar(np.arange(len(v)) + (k - (len(ms) - 1) / 2) * w, v.MAPE, w * 0.88,
                   color=COLOR[m], label=LABEL[m])
        ax.set_xticks(range(3), [f"σ = {s:g}" for s in sorted(tt.noise.unique())])
        ax.set_title(f"{N}³ grid", fontsize=10)
        ax.set_xlabel("noise level")
    axes[0].set_ylabel("mean absolute error (%)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=4, fontsize=8)
    fig.suptitle("Test split: surface-area error vs noise (voxel-face counting, ~50%, omitted)", fontsize=10, y=1.03)
    fig.savefig(os.path.join(root, "figures", "error_vs_noise.png"))
    plt.close(fig)


def fig_signed(df, root):
    """Distribution of signed error per method, test split, both resolutions."""
    ms = [m for m in methods_present(df) if m not in FIG_EXCLUDE]
    d = df[df.split == "test"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, N in zip(axes, (32, 64)):
        dd = d[d.N == N]
        data = [dd[dd.method == m].err.values * 100 for m in ms]
        bp = ax.boxplot(data, widths=0.6, showfliers=False, patch_artist=True,
                        medianprops={"color": "#0b0b0b", "linewidth": 1.2})
        for patch, m in zip(bp["boxes"], ms):
            patch.set_facecolor(COLOR[m]); patch.set_edgecolor(COLOR[m]); patch.set_alpha(0.85)
        ax.axhline(0, color="#52514e", lw=0.8)
        ax.set_xticks(range(1, len(ms) + 1), [LABEL[m] for m in ms], rotation=35, ha="right", fontsize=8)
        ax.set_title(f"{N}³ grid", fontsize=10)
    axes[0].set_ylabel("signed error (%)")
    fig.suptitle("Test split: signed error (boxes: IQR, whiskers: 1.5 IQR)", fontsize=10, y=1.03)
    fig.savefig(os.path.join(root, "figures", "signed_error_box.png"))
    plt.close(fig)


def fig_splits(df, root):
    ms = [m for m in methods_present(df) if m not in FIG_EXCLUDE]
    t = grouped(df, ["split", "N", "method"])
    splits = ["test", "held_super", "held_rough"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    w = 0.8 / len(ms)
    for ax, N in zip(axes, (32, 64)):
        tt = t[t.N == N]
        for k, m in enumerate(ms):
            v = tt[tt.method == m].set_index("split").reindex(splits)
            ax.bar(np.arange(3) + (k - (len(ms) - 1) / 2) * w, v.MAPE, w * 0.88, color=COLOR[m], label=LABEL[m])
        ax.set_xticks(range(3), ["test\n(seen families)", "superellipsoid\n(held out)", "rough blob\n(held out)"], fontsize=8)
        ax.set_title(f"{N}³ grid", fontsize=10)
    axes[0].set_ylabel("mean absolute error (%)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=4, fontsize=8)
    fig.suptitle("Generalisation to unseen shape families (all noise levels pooled)", fontsize=10, y=1.03)
    fig.savefig(os.path.join(root, "figures", "heldout_families.png"))
    plt.close(fig)


def fig_history(root):
    files = sorted(f for f in os.listdir(os.path.join(root, "results")) if f.startswith("history_"))
    if not files:
        return
    fig, ax = plt.subplots(figsize=(5, 3.2))
    for f in files:
        h = pd.read_csv(os.path.join(root, "results", f))
        tag = f[len("history_"):-4]
        m = tag.split("_")[0]
        if m in FIG_EXCLUDE:
            continue
        ls = "-" if tag.endswith("64") else "--"
        ax.plot(h.epoch, h.val_mape, ls, color=COLOR[m], lw=2, label=f"{LABEL[m]}, {tag[-2:]}³")
    ax.set_xlabel("epoch"); ax.set_ylabel("validation MAPE (%)")
    ax.set_yscale("log")
    ax.set_yticks([0.2, 0.3, 0.5, 1, 2, 3], ["0.2", "0.3", "0.5", "1", "2", "3"])
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.legend(fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    fig.savefig(os.path.join(root, "figures", "training_curves.png"))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--from_csv", action="store_true",
                    help="rebuild tables and figures from results/long_errors.csv, without the dataset")
    args = ap.parse_args()
    for d in ("results", "figures"):
        os.makedirs(os.path.join(args.root, d), exist_ok=True)
    if args.from_csv:
        df = pd.read_csv(os.path.join(args.root, "results", "long_errors.csv"))
    else:
        df = long_table(args.root)
        df.to_csv(os.path.join(args.root, "results", "long_errors.csv"), index=False)

    t_split = grouped(df, ["split", "N", "method"])
    t_noise = grouped(df[df.split == "test"], ["N", "noise", "method"])
    t_fam = grouped(df[df.split.isin(["test", "held_super", "held_rough"])], ["family", "N", "method"])
    df["diam_bin"] = pd.cut(df.diam_vox, [0, 16, 24, 32, 48, 64], labels=["<16", "16-24", "24-32", "32-48", "48-64"])
    t_diam = grouped(df[df.split == "test"], ["diam_bin", "method"])
    for name, t in (("split", t_split), ("noise", t_noise), ("family", t_fam), ("diam", t_diam)):
        t.to_csv(os.path.join(args.root, "results", f"metrics_{name}.csv"), index=False)

    md = ["# Results tables\n", "Error = estimated / reference area - 1, in %. MAPE = mean |error|.\n"]
    for split in ("test", "held_super", "held_rough", "val"):
        tt = t_split[t_split.split == split]
        if len(tt):
            md.append(f"\n## {SPLIT_LABEL[split]}: MAPE by resolution\n\n" + fmt_pivot(tt, "N"))
            md.append(f"\n\n## {SPLIT_LABEL[split]}: bias (mean signed error) by resolution\n\n" + fmt_pivot(tt, "N", "bias"))
    md.append("\n\n## Test: MAPE by resolution and noise\n\n" + fmt_pivot(t_noise, ["N", "noise"]))
    md.append("\n\n## Test + held-out: MAPE by family and resolution\n\n" + fmt_pivot(t_fam, ["family", "N"]))
    md.append("\n\n## Test: MAPE by object diameter (voxels)\n\n" + fmt_pivot(t_diam, "diam_bin"))
    md.append("\n\n## Test: R² of area across the split\n\n" + fmt_pivot(t_split[t_split.split == "test"], "N", "R2", 5))

    # best method per regime (lowest MAPE), test + held-out
    reg = grouped(df[df.split.isin(["test", "held_super", "held_rough"])], ["split", "N", "noise", "method"])
    best = reg.loc[reg.groupby(["split", "N", "noise"]).MAPE.idxmin()]
    mcg = reg[reg.method == "mc_grey"].set_index(["split", "N", "noise"]).MAPE
    best = best.assign(best=best.method.map(LABEL), MC_grey_MAPE=mcg.reindex(
        list(zip(best.split, best.N, best.noise))).values)
    md.append("\n\n## Best estimator per regime\n\n" +
              best[["split", "N", "noise", "best", "MAPE", "MC_grey_MAPE"]].round(2).to_markdown(index=False))
    with open(os.path.join(args.root, "results", "tables.md"), "w") as f:
        f.write("\n".join(md) + "\n")

    fig_noise(df, args.root)
    fig_signed(df, args.root)
    fig_splits(df, args.root)
    fig_history(args.root)
    print("\n".join(md[:4]))
    print("wrote results/tables.md and figures/")


if __name__ == "__main__":
    main()
