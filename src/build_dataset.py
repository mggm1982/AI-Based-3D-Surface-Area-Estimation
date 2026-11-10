"""
Generate the full dataset.

Each shape instance is rendered at N = 32 and 64 and at three noise levels,
so one shape gives 6 samples. Splits are made by shape instance, so all six
variants of a shape always land in the same split (no leakage).

Splits
  train / val / test : shapes from the training families (70/15/15 per family)
  held_super          : superellipsoids only (family never seen in training)
  held_rough          : rough blobs only (roughness never seen in training)

Output layout
  training_data/  vol_{train,val}_N{32,64}.npy   uint8 volumes (M, N, N, N)
                  pts_{train,val}_N{32,64}.npy   float16 MC surface points (M, 2048, 3)
                  labels_{train,val}.csv
  testing_data/   same for test, held_super, held_rough, plus example_meshes/

Usage
  python src/build_dataset.py --out . --workers 4            # full dataset
  python src/build_dataset.py --out . --workers 4 --resume   # continue after an interruption
  python src/build_dataset.py --out /tmp/small --scale 0.02  # quick test
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from classical import all_estimates, mc_mesh, sample_surface_points
from shapes import PlacedShape, make_shape, random_rotation
from voxelize import NOISE_LEVELS, ct_image, decode, encode, occupancy

SEED = 51
RESOLUTIONS = (32, 64)
N_POINTS = 2048
FILL_RANGE = (0.35, 1.0)

FAMILY_COUNTS = {"sphere": 150, "ellipsoid": 350, "cylinder": 300, "torus": 300, "blob": 900}
HELDOUT_COUNTS = {"held_super": ("superellipsoid", 200), "held_rough": ("rough_blob", 200)}
SPLIT_FRAC = (0.70, 0.15, 0.15)
SPLIT_DIR = {"train": "training_data", "val": "training_data", "test": "testing_data",
             "held_super": "testing_data", "held_rough": "testing_data"}


def plan_shapes(scale: float = 1.0) -> list[dict]:
    """Assign every shape instance a global id, family and split."""
    rng = np.random.default_rng(SEED)
    jobs, sid = [], 0
    for fam, n in FAMILY_COUNTS.items():
        n = max(3, int(round(n * scale)))
        perm = rng.permutation(n)
        n_tr, n_va = int(round(SPLIT_FRAC[0] * n)), int(round(SPLIT_FRAC[1] * n))
        for k in range(n):
            split = "train" if perm[k] < n_tr else ("val" if perm[k] < n_tr + n_va else "test")
            jobs.append({"shape_id": sid, "family": fam, "split": split})
            sid += 1
    for split, (fam, n) in HELDOUT_COUNTS.items():
        for _ in range(max(2, int(round(n * scale)))):
            jobs.append({"shape_id": sid, "family": fam, "split": split})
            sid += 1
    # position of each shape inside its split
    counters: dict[str, int] = {}
    for j in jobs:
        j["slot"] = counters.get(j["split"], 0)
        counters[j["split"]] = j["slot"] + 1
    return jobs


def render_shape(job: dict) -> dict:
    rng = np.random.default_rng([SEED, job["shape_id"]])
    shape = make_shape(job["family"], rng)
    fill = rng.uniform(*FILL_RANGE)
    rot = random_rotation(rng)
    rows, vols, pts = [], {}, {}
    for N in RESOLUTIONS:
        ps = PlacedShape(shape, N, fill, rot, rng.uniform(-0.5, 0.5, 3))
        a_true, a_err = ps.area()
        occ = occupancy(ps)
        vols[N], pts[N] = [], []
        for j, sn in enumerate(NOISE_LEVELS):
            nrng = np.random.default_rng([SEED, job["shape_id"], N, j])
            q = encode(ct_image(occ, sn, nrng))
            img = decode(q)
            est = all_estimates(img)
            verts, faces = mc_mesh(img)
            p = sample_surface_points(verts, faces, N_POINTS, nrng)
            vols[N].append(q)
            pts[N].append(p.astype(np.float16))
            rows.append({
                "shape_id": job["shape_id"], "family": job["family"], "split": job["split"],
                "N": N, "noise": sn, "noise_idx": j, "fill": fill,
                "diam_vox": 2 * ps.scale, "A_true": a_true, "A_ref_err": a_err,
                "V_true": ps.volume(), "V_occ_clean": float(occ.sum()),
                **est, "params": json.dumps(shape.params()),
            })
    return {"job": job, "rows": rows, "vols": vols, "pts": pts}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=".")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--scale", type=float, default=1.0, help="fraction of the full shape counts")
    ap.add_argument("--resume", action="store_true", help="continue an interrupted run")
    args = ap.parse_args()

    jobs = plan_shapes(args.scale)
    n_per_split = pd.Series([j["split"] for j in jobs]).value_counts().to_dict()
    nL = len(NOISE_LEVELS)
    for d in set(SPLIT_DIR.values()):
        os.makedirs(os.path.join(args.out, d), exist_ok=True)

    # Resume support: finished shapes are logged to progress.jsonl, so an
    # interrupted run continues where it stopped when started with --resume.
    prog_path = os.path.join(args.out, SPLIT_DIR["train"], "progress.jsonl")
    all_rows, done = [], set()
    if args.resume and os.path.exists(prog_path):
        with open(prog_path) as f:
            for line in f:
                rec = json.loads(line)
                done.add(rec["shape_id"])
                all_rows.extend(rec["rows"])
        print(f"resuming: {len(done)} shapes already done", flush=True)
    mode = "r+" if done else "w+"

    mm = {}
    for split, n in n_per_split.items():
        d = os.path.join(args.out, SPLIT_DIR[split])
        for N in RESOLUTIONS:
            p_vol, p_pts = os.path.join(d, f"vol_{split}_N{N}.npy"), os.path.join(d, f"pts_{split}_N{N}.npy")
            if mode == "r+":
                mm[("vol", split, N)] = np.load(p_vol, mmap_mode="r+")
                mm[("pts", split, N)] = np.load(p_pts, mmap_mode="r+")
            else:
                mm[("vol", split, N)] = np.lib.format.open_memmap(p_vol, mode="w+", dtype=np.uint8,
                                                                  shape=(n * nL, N, N, N))
                mm[("pts", split, N)] = np.lib.format.open_memmap(p_pts, mode="w+", dtype=np.float16,
                                                                  shape=(n * nL, N_POINTS, 3))
    if mode == "w+" and os.path.exists(prog_path):
        os.remove(prog_path)

    todo = [j for j in jobs if j["shape_id"] not in done]
    t0 = time.time()
    # ProcessPoolExecutor raises an error if a worker dies (e.g. out of memory)
    # instead of hanging forever like multiprocessing.Pool.
    with ProcessPoolExecutor(args.workers) as ex, open(prog_path, "a") as prog:
        futures = [ex.submit(render_shape, j) for j in todo]
        for k, fut in enumerate(as_completed(futures)):
            res = fut.result()
            job = res["job"]
            for N in RESOLUTIONS:
                for j in range(nL):
                    idx = job["slot"] * nL + j
                    mm[("vol", job["split"], N)][idx] = res["vols"][N][j]
                    mm[("pts", job["split"], N)][idx] = res["pts"][N][j]
            for r in res["rows"]:
                r["index"] = job["slot"] * nL + r["noise_idx"]
            all_rows.extend(res["rows"])
            if (k + 1) % 25 == 0:
                for m in mm.values():
                    m.flush()
            prog.write(json.dumps({"shape_id": job["shape_id"], "rows": res["rows"]}) + "\n")
            prog.flush()
            if (k + 1) % 50 == 0 or k + 1 == len(todo):
                el = time.time() - t0
                print(f"{len(done) + k + 1}/{len(jobs)} shapes  {el / 60:.1f} min  "
                      f"eta {el / (k + 1) * (len(todo) - k - 1) / 60:.1f} min", flush=True)

    for m in mm.values():
        m.flush()
    df = pd.DataFrame(all_rows).sort_values(["split", "N", "index"])
    for split in n_per_split:
        df[df.split == split].to_csv(os.path.join(args.out, SPLIT_DIR[split], f"labels_{split}.csv"), index=False)
    print("done", len(df), "samples")


if __name__ == "__main__":
    main()
