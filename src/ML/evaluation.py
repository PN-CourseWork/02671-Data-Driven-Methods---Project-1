"""Evaluation and rollout utilities for trained PDE time-stepping models."""

import numpy as np
import torch


def rollout_trajectory(model, u0, n_steps, device):
    """
    Autoregressively advance the model from u0.

    Parameters
    ----------
    model : nn.Module
        Trained time-stepper (must have a .rollout method).
    u0 : Tensor, shape (input_dim,) or (1, input_dim)
        Initial condition (normalized).
    n_steps : int
        Number of steps to advance.
    device : torch.device

    Returns
    -------
    ndarray, shape (n_steps + 1, input_dim)
        Predicted trajectory including IC.
    """
    model.eval()
    if u0.dim() == 1:
        u0 = u0.unsqueeze(0)
    return model.rollout(u0.to(device), n_steps).cpu().numpy()


def compute_rollout_error(traj_pred, traj_true):
    """
    Compute relative L2 error at each timestep (vectorized).

    Returns ndarray of shape (min(T_pred, T_true),).
    """
    n = min(len(traj_pred), len(traj_true))
    norms = np.linalg.norm(traj_true[:n], axis=1)
    diffs = np.linalg.norm(traj_pred[:n] - traj_true[:n], axis=1)
    # Avoid division by zero
    norms = np.where(norms > 0, norms, 1.0)
    return diffs / norms


def one_step_predictions(model, dataset, indices, device):
    """
    Compute one-step predictions for given snapshot indices.

    Returns list of dicts with keys "time", "true", "pred" (physical space).
    """
    model.eval()
    results = []
    for idx in indices:
        u_t = (
            torch.from_numpy((dataset.snapshots[idx] - dataset.mean) / dataset.std)
            .float()
            .unsqueeze(0)
            .to(device)
        )
        with torch.no_grad():
            pred_norm = model(u_t).cpu().numpy().squeeze()

        results.append(
            {
                "time": dataset.times[idx],
                "true": dataset.snapshots[idx + 1],
                "pred": pred_norm * dataset.std + dataset.mean,
            }
        )
    return results
