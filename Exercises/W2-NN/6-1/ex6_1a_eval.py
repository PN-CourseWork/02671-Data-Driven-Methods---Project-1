# Exercise 6.1a — Post-training evaluation for KS MLP time-stepper
#
# Loads the best checkpoint from train.py and produces:
#   - One-step prediction comparison plots
#   - Autoregressive rollout vs ground truth
#   - Rollout error growth curve
#
# The model architecture is reconstructed from the config saved in the
# checkpoint — no need to hardcode hidden_dims or class names.
#
# Usage:
#   uv run python Exercises/W2-NN/ex6_1a_eval.py
#   uv run python Exercises/W2-NN/ex6_1a_eval.py --checkpoint path/to/model.pt

import argparse
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch

from ML import evaluation, plotting, trainer
from ML.dataset import KSDataset

plt.style.use("./science.mplstyle")

# =============================================================================
# Configuration
# =============================================================================

parser = argparse.ArgumentParser(description="Evaluate a trained KS time-stepper.")
parser.add_argument(
    "--checkpoint",
    type=Path,
    default=Path("checkpoints/best_model.pt"),
    help="Path to model checkpoint.",
)
args = parser.parse_args()

CHECKPOINT = args.checkpoint
FIGURES_DIR = Path("figures/ex6-1a")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Load model and data
# =============================================================================

device = trainer.get_device()
dataset = KSDataset(data_dir="DATA/W2-NN/KS", normalize=True)

# Load checkpoint and reconstruct model from saved config
checkpoint = trainer.load_checkpoint(CHECKPOINT, device)
model = hydra.utils.instantiate(checkpoint["cfg"]["model"]).to(device)
model.load_state_dict(checkpoint["model_state_dict"])

print(f"Loaded checkpoint: epoch {checkpoint['epoch']}, val loss {checkpoint['loss']:.6f}")
print(f"Model config: {checkpoint['cfg']['model']}")
print(f"Dataset: {len(dataset)} samples, dt = {dataset.dt:.4f}")

# =============================================================================
# One-step predictions
# =============================================================================

rng = np.random.default_rng(0)
indices = rng.choice(len(dataset) - 1, size=4, replace=False)
samples = evaluation.one_step_predictions(model, dataset, indices, device)
plotting.plot_one_step_samples(samples, dataset.x, FIGURES_DIR)
print(f"Saved one-step predictions to {FIGURES_DIR}")

# =============================================================================
# Autoregressive rollout
# =============================================================================

n_rollout = min(200, len(dataset.snapshots) - 1)
u0_norm = torch.from_numpy((dataset.snapshots[0] - dataset.mean) / dataset.std).float()
traj_pred = dataset.denormalize(evaluation.rollout_trajectory(model, u0_norm, n_rollout, device))
traj_true = dataset.snapshots[: n_rollout + 1]
times = dataset.times[: n_rollout + 1]

plotting.plot_rollout_comparison(traj_true, traj_pred, times, dataset.x, FIGURES_DIR)
print(f"Saved rollout comparison to {FIGURES_DIR}")

# =============================================================================
# Rollout error
# =============================================================================

errors = evaluation.compute_rollout_error(traj_pred, traj_true)
plotting.plot_rollout_error(errors, times, FIGURES_DIR)

print(f"\nRollout error — mean: {errors.mean():.4f}, final: {errors[-1]:.4f}")
