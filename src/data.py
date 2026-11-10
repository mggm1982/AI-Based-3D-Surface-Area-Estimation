"""Loading the generated dataset and building model inputs/targets."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch

SPLIT_DIR = {"train": "training_data", "val": "training_data", "test": "testing_data",
             "held_super": "testing_data", "held_rough": "testing_data"}
EVAL_SPLITS = ("val", "test", "held_super", "held_rough")


def load_labels(root: str, split: str, N: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(root, SPLIT_DIR[split], f"labels_{split}.csv"))
    if N is not None:
        df = df[df.N == N].sort_values("index").reset_index(drop=True)
    return df


def load_array(root: str, kind: str, split: str, N: int, mmap: bool = False) -> np.ndarray:
    path = os.path.join(root, SPLIT_DIR[split], f"{kind}_{split}_N{N}.npy")
    return np.load(path, mmap_mode="r" if mmap else None)


def target(df: pd.DataFrame) -> np.ndarray:
    """y = log(A_true / A_mc_grey)."""
    return np.log(df.A_true.values / df.A_mc_grey.values).astype(np.float32)


def volumes_to_tensor(q: np.ndarray, device) -> torch.Tensor:
    """uint8 (B, N, N, N) -> float (B, 1, N, N, N), background -1, object +1."""
    x = torch.from_numpy(np.ascontiguousarray(q)).to(device).float()
    x = (x / 127.5 - 0.5 - 0.5) * 2.0
    return x.unsqueeze(1)


def random_cube_symmetry(x: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """One of the 48 symmetries of the cube (axis permutation + flips), applied to
    the whole batch. Exact for voxel grids: no interpolation, area unchanged."""
    perm = torch.randperm(3, generator=gen).tolist()
    x = x.permute(0, 1, *[p + 2 for p in perm])
    flips = [d + 2 for d in range(3) if torch.rand(1, generator=gen).item() < 0.5]
    return torch.flip(x, flips) if flips else x


def points_to_tensor(p: np.ndarray, N: int, device) -> torch.Tensor:
    """Centre on the grid centre; divide by a fixed constant (not per sample) so
    the voxel size, and hence the scale of the shape, is preserved."""
    x = torch.from_numpy(np.asarray(p, dtype=np.float32)).to(device)
    return (x - (N - 1) / 2.0) / 32.0


def random_rotations(B: int, gen: torch.Generator, device) -> torch.Tensor:
    """B uniformly random rotation matrices (QR of Gaussian matrices)."""
    a = torch.randn(B, 3, 3, generator=gen)
    q, r = torch.linalg.qr(a)
    q = q * torch.sign(torch.diagonal(r, dim1=1, dim2=2)).unsqueeze(1)
    det = torch.det(q)
    q[:, :, 0] *= det.unsqueeze(1)
    return q.to(device)
