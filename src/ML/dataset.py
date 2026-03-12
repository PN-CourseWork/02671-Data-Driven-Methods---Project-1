"""PyTorch Dataset classes for PDE time-stepping data."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class KSDataset(Dataset):
    """
    Dataset of consecutive KS snapshot pairs (u_t, u_{t+dt}).

    Parameters
    ----------
    data_dir : str or Path
        Directory containing ks_uu.npy, ks_tt.npy, and ks_x.npy.
    normalize : bool
        Whether to normalize data to zero mean, unit std.
    t_start, t_end : float or None
        Optional time window for data selection.
    """

    def __init__(self, data_dir="DATA/W2-NN/KS", normalize=True, t_start=None, t_end=None):
        data_dir = Path(data_dir)
        uu = np.load(data_dir / "ks_uu.npy")  # (N, n_times)
        tt = np.load(data_dir / "ks_tt.npy")  # (n_times,)

        # Select time window
        mask = np.ones(len(tt), dtype=bool)
        if t_start is not None:
            mask &= tt >= t_start
        if t_end is not None:
            mask &= tt <= t_end
        uu = uu[:, mask]
        tt = tt[mask]

        self.snapshots = uu.T.astype(np.float32)  # (n_times, N)
        self.times = tt
        self.dt = tt[1] - tt[0]
        self.x = np.load(data_dir / "ks_x.npy")  # (N,) spatial grid
        self.data_dir = data_dir

        # Normalization
        if normalize:
            self.mean = self.snapshots.mean()
            self.std = self.snapshots.std()
        else:
            self.mean = 0.0
            self.std = 1.0

    def __len__(self):
        return len(self.snapshots) - 1

    def __getitem__(self, idx):
        u_t = (self.snapshots[idx] - self.mean) / self.std
        u_next = (self.snapshots[idx + 1] - self.mean) / self.std
        return torch.from_numpy(u_t), torch.from_numpy(u_next)

    def denormalize(self, x):
        """Undo normalization."""
        return x * self.std + self.mean
