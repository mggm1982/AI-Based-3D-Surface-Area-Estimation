"""
Calibrated classical baselines, fit on the training split only.

  lookup : A_mc_grey * exp(median y) per (resolution, noise level). This is the
           "learn one constant per regime" baseline. It is given the true noise
           level, so it is optimistic; a learned model has to beat it to show
           that it does more than memorise a regime-dependent bias.
  rf     : Random Forest on hand-crafted descriptors (classical areas, volume,
           aspect ratios, Euler number, estimated noise), predicting the same
           log-ratio target as the networks.

  python src/baselines.py      -> results/preds_lookup.csv, results/preds_rf.csv
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from data import EVAL_SPLITS, load_labels, target


def features(df: pd.DataFrame) -> np.ndarray:
    g = df.A_mc_grey.values

    def lr(a):
        # log-ratio to MC-grey; smoothing can erase very thin objects (area 0),
        # so the ratio is floored at 1e-3 instead of giving log(0)
        return np.log(np.maximum(a / g, 1e-3))

    return np.column_stack([
        np.log(g / df.N.values ** 2),
        lr(df.A_face.values),
        lr(df.A_mc_bin.values),
        lr(df.A_mc_smooth.values),
        np.log(np.maximum(df.vol_occ.values, 1.0) ** (2 / 3) / g),   # compactness
        df.aspect1.values, df.aspect2.values,
        np.clip(df.euler.values, -50, 200),
        df.noise_hat.values,
        df.N.values,
    ])


FEATURE_NAMES = ["log_mcgrey_over_N2", "log_face_over_mcgrey", "log_mcbin_over_mcgrey",
                 "log_mcsmooth_over_mcgrey", "log_compactness", "aspect1", "aspect2",
                 "euler", "noise_hat", "N"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    args = ap.parse_args()
    os.makedirs(os.path.join(args.root, "results"), exist_ok=True)

    tr = load_labels(args.root, "train")
    y = target(tr)
    lut = pd.Series(y).groupby([tr.N.values, tr.noise_idx.values]).median()
    rf = RandomForestRegressor(n_estimators=400, min_samples_leaf=3, max_features=0.5,
                               n_jobs=-1, random_state=51)
    rf.fit(features(tr), y)
    imp = pd.Series(rf.feature_importances_, index=FEATURE_NAMES).sort_values(ascending=False)
    imp.to_csv(os.path.join(args.root, "results", "rf_feature_importance.csv"))
    print("RF feature importance\n", imp.round(3).to_string())

    out_l, out_r = [], []
    for split in EVAL_SPLITS:
        df = load_labels(args.root, split)
        y_l = np.array([lut[(n, k)] for n, k in zip(df.N, df.noise_idx)])
        y_r = rf.predict(features(df))
        base = dict(split=split, N=df.N.values, index=df["index"].values, shape_id=df.shape_id.values)
        out_l.append(pd.DataFrame({**base, "y_pred": y_l, "A_pred": df.A_mc_grey.values * np.exp(y_l)}))
        out_r.append(pd.DataFrame({**base, "y_pred": y_r, "A_pred": df.A_mc_grey.values * np.exp(y_r)}))
        for name, yy in (("lookup", y_l), ("rf", y_r)):
            e = df.A_mc_grey.values * np.exp(yy) / df.A_true.values - 1
            print(f"{split:11s} {name:7s} MAPE {100 * np.mean(np.abs(e)):.2f}%")
    pd.concat(out_l).to_csv(os.path.join(args.root, "results", "preds_lookup.csv"), index=False)
    pd.concat(out_r).to_csv(os.path.join(args.root, "results", "preds_rf.csv"), index=False)


if __name__ == "__main__":
    main()
