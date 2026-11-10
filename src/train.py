"""
Train a CNN3D or PointNet surface-area corrector at one resolution.

  python src/train.py --model cnn --N 32
  python src/train.py --model cnn --N 64
  python src/train.py --model pointnet --N 32
  python src/train.py --model pointnet --N 64

Outputs
  weights/{model}_N{N}.pt               best weights (lowest validation MAPE)
  results/history_{model}_N{N}.csv      per-epoch training curve
  results/preds_{model}_N{N}.csv        predictions on val, test and held-out splits
"""
from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from data import (EVAL_SPLITS, load_array, load_labels, points_to_tensor, random_cube_symmetry,
                  random_rotations, target, volumes_to_tensor)
from models import CNN3D, PointNet, count_params

Y_SCALE = 0.05      # targets are divided by this so the Huber threshold (1.0) is a 5% error


def pick_device(name: str):
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def make_batch(kind, arr, idx, N, device, gen, augment):
    if kind == "cnn":
        x = volumes_to_tensor(arr[idx], device)
        return random_cube_symmetry(x, gen) if augment else x
    x = points_to_tensor(arr[idx], N, device)
    if augment:
        x = x @ random_rotations(len(idx), gen, device)
    return x


@torch.no_grad()
def predict(model, kind, arr, N, device, bs=64):
    model.eval()
    out = []
    for i in range(0, len(arr), bs):
        idx = np.arange(i, min(i + bs, len(arr)))
        out.append(model(make_batch(kind, arr, idx, N, device, None, False)).float().cpu().numpy())
    return np.concatenate(out) * Y_SCALE


def mape(y_pred, df):
    a_pred = df.A_mc_grey.values * np.exp(y_pred)
    return float(np.mean(np.abs(a_pred / df.A_true.values - 1)) * 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["cnn", "pointnet"], required=True)
    ap.add_argument("--N", type=int, choices=[32, 64], required=True)
    ap.add_argument("--root", default=".")
    ap.add_argument("--epochs", type=int, default=None,
                    help="default 40; 15 for the 64^3 CNN (about 8x the cost per epoch)")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--limit", type=int, default=0, help="debug: use only this many training samples")
    ap.add_argument("--seed", type=int, default=51)
    args = ap.parse_args()

    if args.epochs is None:
        args.epochs = 15 if (args.model == "cnn" and args.N == 64) else 40
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device(args.device)
    kind, N = args.model, args.N
    akind = "pts" if kind == "pointnet" else "vol"
    os.makedirs(os.path.join(args.root, "weights"), exist_ok=True)
    os.makedirs(os.path.join(args.root, "results"), exist_ok=True)
    tag = f"{kind}_N{N}"

    df_tr = load_labels(args.root, "train", N)
    x_tr = load_array(args.root, akind, "train", N)          # loaded fully into RAM
    if args.limit:
        df_tr, x_tr = df_tr.iloc[:args.limit], x_tr[:args.limit]
    y_tr = torch.from_numpy(target(df_tr) / Y_SCALE)
    df_va = load_labels(args.root, "val", N)
    x_va = load_array(args.root, akind, "val", N)

    model = (PointNet() if kind == "pointnet" else CNN3D()).to(device)
    print(f"{tag}: {count_params(model):,} params, device {device}, "
          f"{len(df_tr)} train / {len(df_va)} val samples", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * int(np.ceil(len(df_tr) / args.bs)), pct_start=0.15)
    loss_fn = nn.HuberLoss(delta=1.0)
    gen = torch.Generator().manual_seed(args.seed)

    base_mape = mape(np.zeros(len(df_va)), df_va)
    print(f"val MAPE of uncorrected MC (grey): {base_mape:.2f}%", flush=True)
    best, best_ep, hist = np.inf, -1, []
    wpath = os.path.join(args.root, "weights", f"{tag}.pt")
    for ep in range(args.epochs):
        model.train()
        t0, tot, nb = time.time(), 0.0, 0
        perm = torch.randperm(len(df_tr), generator=gen).numpy()
        for i in range(0, len(perm), args.bs):
            idx = np.sort(perm[i:i + args.bs])
            if len(idx) < 2:
                continue
            xb = make_batch(kind, x_tr, idx, N, device, gen, True)
            yb = y_tr[idx].to(device)
            loss = loss_fn(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot, nb = tot + loss.item(), nb + 1
        va = mape(predict(model, kind, x_va, N, device), df_va)
        hist.append({"epoch": ep + 1, "train_loss": tot / nb, "val_mape": va, "sec": time.time() - t0})
        flag = ""
        if va < best:
            best, best_ep, flag = va, ep, " *"
            torch.save(model.state_dict(), wpath)
        print(f"epoch {ep + 1:3d}  loss {tot / nb:.4f}  val MAPE {va:.3f}%  ({time.time() - t0:.0f}s){flag}", flush=True)
        pd.DataFrame(hist).to_csv(os.path.join(args.root, "results", f"history_{tag}.csv"), index=False)
        if ep - best_ep >= args.patience:
            print("early stopping", flush=True)
            break

    model.load_state_dict(torch.load(wpath, map_location=device, weights_only=True))
    rows = []
    for split in EVAL_SPLITS:
        try:
            df = load_labels(args.root, split, N)
            x = load_array(args.root, akind, split, N, mmap=True)
        except FileNotFoundError:
            continue
        y = predict(model, kind, x, N, device)
        rows.append(pd.DataFrame({"split": split, "N": N, "index": df["index"], "shape_id": df.shape_id,
                                  "y_pred": y, "A_pred": df.A_mc_grey.values * np.exp(y)}))
        print(f"{split:11s} MAPE  MC-grey {mape(np.zeros(len(df)), df):.2f}%  ->  {tag} {mape(y, df):.2f}%")
    pd.concat(rows).to_csv(os.path.join(args.root, "results", f"preds_{tag}.csv"), index=False)
    print(f"best val MAPE {best:.3f}% at epoch {best_ep + 1}; weights -> {wpath}")


if __name__ == "__main__":
    main()
