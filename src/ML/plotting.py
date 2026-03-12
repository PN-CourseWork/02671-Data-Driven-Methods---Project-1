"""
Reusable plotting functions for PDE time-stepping experiments.

All plot_* functions save a PDF and return the figure for wandb logging.
"""

from contextlib import contextmanager
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


@contextmanager
def _savefig(figures_dir, filename):
    """Context manager that creates a figure, yields it, then saves + closes."""
    plt.style.use("./science.mplstyle")
    fig = plt.figure()
    yield fig
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / filename, bbox_inches="tight")
    plt.close(fig)


def plot_loss_curves(train_losses, val_losses, figures_dir):
    """Plot training and validation loss curves."""
    with _savefig(figures_dir, "loss_curves.pdf") as fig:
        ax = fig.add_subplot(111)
        epochs = np.arange(1, len(train_losses) + 1)
        ax.semilogy(epochs, train_losses, label="Train")
        ax.semilogy(epochs, val_losses, label="Validation")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.set_title("Training Loss Curves")
        ax.legend()
    return fig


def plot_rollout_comparison(
    traj_true, traj_pred, times, x, figures_dir, filename="rollout_comparison.pdf"
):
    """Side-by-side pcolor of ground truth vs NN rollout."""
    plt.style.use("./science.mplstyle")
    n_show = min(len(traj_true), len(traj_pred))
    T, X = np.meshgrid(times[:n_show], x)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, data, title in zip(
        axes,
        [traj_true[:n_show], traj_pred[:n_show]],
        ["Ground truth", "NN rollout"],
        strict=True,
    ):
        pcm = ax.pcolormesh(T, X, data.T, shading="auto", cmap="hot")
        fig.colorbar(pcm, ax=ax)
        ax.set(title=title, xlabel="Time $t$", ylabel="Space $x$")

    fig.tight_layout()
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / filename, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_one_step_samples(results, x, figures_dir, filename="one_step_predictions.pdf"):
    """Plot one-step predictions vs ground truth for a list of sample dicts."""
    plt.style.use("./science.mplstyle")
    n = len(results)
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.5 * n), squeeze=False)

    for ax, res in zip(axes.flat, results, strict=True):
        ax.plot(x, res["true"], "k-", lw=0.8, label="True $u(t+\\Delta t)$")
        ax.plot(x, res["pred"], "r--", lw=0.8, label="NN prediction")
        ax.set_title(f"$t = {res['time']:.2f}$")
        ax.legend(fontsize=8)
    axes[-1, 0].set_xlabel("$x$")

    fig.tight_layout()
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / filename, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_rollout_error(errors, times, figures_dir, filename="rollout_error.pdf"):
    """Plot relative rollout error vs time."""
    with _savefig(figures_dir, filename) as fig:
        ax = fig.add_subplot(111)
        ax.semilogy(times[: len(errors)], errors)
        ax.set_xlabel("Time $t$")
        ax.set_ylabel("Relative $L_2$ error")
        ax.set_title("Rollout Error Growth")
    return fig
