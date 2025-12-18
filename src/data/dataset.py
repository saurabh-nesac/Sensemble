# src/data/dataset.py

from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from netCDF4 import Dataset as NC


class WRFDAEnsembleICDataset(Dataset):
    """
    Dataset for WRFDA ensemble initial conditions (wrfarw.memXXX).

    Each ensemble member = one sample
    No time dimension is used (Time=1 always)

    Returns:
        cond  : conditioning fields  [C_cond, H, W]
        target: generated fields     [C_tgt,  H, W]
    """

    def __init__(
            self,
            ens_dir: str,
            conditioning_vars,
            target_vars,
            patch_size: int = 64,
            normalize: str = "per_variable",
            augment_factor: int = 50,
            level: int | None = 0,  # 👈 ADD THIS
    ):

        self.ens_dir = Path(ens_dir)
        self.cond_vars = conditioning_vars
        self.tgt_vars = target_vars
        self.patch = patch_size
        self.augment = augment_factor

        self.files = sorted(self.ens_dir.glob("wrfarw.mem*"))
        if len(self.files) == 0:
            raise RuntimeError("No wrfarw.memXXX files found")

        # --------------------------------------------------
        # Load all members into memory (ICs are static)
        # --------------------------------------------------
        cond_list = []
        tgt_list = []

        for f in self.files:
            with NC(f) as ds:
                cond = []
                for v in self.cond_vars:
                    var = ds[v]

                    if var.ndim == 4:
                        # (Time, bottom_top, south_north, west_east)
                        if level is None:
                            raise ValueError(f"Variable {v} is 3D but level=None")
                        arr = var[0, level]  # 👈 FIXED vertical slice
                    elif var.ndim == 3:
                        # (Time, south_north, west_east)
                        arr = var[0]
                    else:
                        raise ValueError(f"Unsupported variable shape for {v}")
                    # Time=0
                    cond.append(arr.astype(np.float32))

                tgt = []
                for v in self.tgt_vars:
                    var = ds[v]

                    if var.ndim == 4:
                        # (Time, bottom_top, south_north, west_east)
                        if level is None:
                            raise ValueError(f"Variable {v} is 3D but level=None")
                        arr = var[0, level]  # 👈 FIXED vertical slice
                    elif var.ndim == 3:
                        # (Time, south_north, west_east)
                        arr = var[0]
                    else:
                        raise ValueError(f"Unsupported variable shape for {v}")
                    # Time=0
                    tgt.append(arr.astype(np.float32))

            cond_list.append(np.stack(cond, axis=0))
            tgt_list.append(np.stack(tgt, axis=0))

        # Shapes:
        # cond: [M, Cc, H, W]
        # tgt : [M, Ct, H, W]
        self.cond = np.stack(cond_list, axis=0)
        self.tgt = np.stack(tgt_list, axis=0)

        self.M, self.Cc, self.H, self.W = self.cond.shape
        self.Ct = self.tgt.shape[1]

        # --------------------------------------------------
        # Normalization (CRITICAL for GAN stability)
        # --------------------------------------------------
        if normalize == "per_variable":
            self._normalize_per_variable()
        elif normalize == "global":
            self._normalize_global()
        elif normalize is None:
            pass
        else:
            raise ValueError(f"Unknown normalize option: {normalize}")

    # ======================================================
    # Normalization methods
    # ======================================================
    def _normalize_per_variable(self):
        for c in range(self.Cc):
            mu = self.cond[:, c].mean()
            std = self.cond[:, c].std() + 1e-6
            self.cond[:, c] = (self.cond[:, c] - mu) / std

        for c in range(self.Ct):
            mu = self.tgt[:, c].mean()
            std = self.tgt[:, c].std() + 1e-6
            self.tgt[:, c] = (self.tgt[:, c] - mu) / std

    def _normalize_global(self):
        mu = self.cond.mean()
        std = self.cond.std() + 1e-6
        self.cond = (self.cond - mu) / std

        mu = self.tgt.mean()
        std = self.tgt.std() + 1e-6
        self.tgt = (self.tgt - mu) / std

    # ======================================================
    # PyTorch API
    # ======================================================
    def __len__(self):
        # artificial enlargement via random cropping
        return self.M * self.augment

    def __getitem__(self, idx):
        # pick ensemble member
        m = random.randint(0, self.M - 1)

        # random spatial crop
        y0 = random.randint(0, self.H - self.patch)
        x0 = random.randint(0, self.W - self.patch)

        cond_patch = self.cond[
            m, :, y0 : y0 + self.patch, x0 : x0 + self.patch
        ]

        tgt_patch = self.tgt[
            m, :, y0 : y0 + self.patch, x0 : x0 + self.patch
        ]

        return (
            torch.from_numpy(cond_patch),
            torch.from_numpy(tgt_patch),
        )
