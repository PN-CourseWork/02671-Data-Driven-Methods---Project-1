"""
Generic training loop for PDE time-stepping models.

Used by train.py — the single Hydra entry point.
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader


def get_device():
    """
    Select the best available device: CUDA > MPS (Apple Silicon) > CPU.

    Returns
    -------
    torch.device
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_epoch(model, dataloader, optimizer, criterion, device):
    """
    Train for one epoch.

    Returns
    -------
    float
        Mean training loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for u_t, u_next in dataloader:
        u_t = u_t.to(device)
        u_next = u_next.to(device)

        optimizer.zero_grad()
        pred = model(u_t)
        loss = criterion(pred, u_next)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def eval_epoch(model, dataloader, criterion, device):
    """
    Evaluate on a dataset.

    Returns
    -------
    float
        Mean evaluation loss.
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for u_t, u_next in dataloader:
            u_t = u_t.to(device)
            u_next = u_next.to(device)

            pred = model(u_t)
            loss = criterion(pred, u_next)

            total_loss += loss.item()
            n_batches += 1

    return total_loss / n_batches


def save_checkpoint(model, optimizer, epoch, loss, path, cfg=None):
    """
    Save a model checkpoint.

    Parameters
    ----------
    model : nn.Module
    optimizer : Optimizer
    epoch : int
    loss : float
    path : str or Path
    cfg : dict or None
        Resolved Hydra config to store alongside weights so eval scripts
        can reconstruct the model without hardcoding architecture params.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }
    if cfg is not None:
        payload["cfg"] = cfg
    torch.save(payload, path)
    return path


def load_checkpoint(path, device, model=None, optimizer=None):
    """
    Load a model checkpoint.

    If *model* is None, returns the raw checkpoint dict so the caller can
    reconstruct the model from ``checkpoint["cfg"]`` first.

    Parameters
    ----------
    path : str or Path
    device : torch.device
    model : nn.Module or None
    optimizer : Optimizer or None

    Returns
    -------
    dict
        The full checkpoint dict.  Always contains "epoch", "loss", and
        "model_state_dict".  When a config was saved it also has "cfg".
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if model is not None:
        model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


def count_parameters(model):
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make_dataloaders(dataset, train_frac, batch_size, seed=42):
    """
    Split a dataset into train/val and return DataLoaders.

    Parameters
    ----------
    dataset : Dataset
    train_frac : float
    batch_size : int
    seed : int

    Returns
    -------
    train_loader, val_loader : DataLoader
    """
    n_total = len(dataset)
    n_train = int(train_frac * n_total)
    n_val = n_total - n_train

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    print(f"Training samples: {n_train}, Validation samples: {n_val}")
    return train_loader, val_loader


def fit(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    criterion,
    device,
    epochs,
    log_every=5,
    figures_dir=None,
    log_fn=None,
    cfg=None,
):
    """
    Full training loop.

    Parameters
    ----------
    model : nn.Module
    train_loader, val_loader : DataLoader
    optimizer : Optimizer
    scheduler : LR scheduler
    criterion : loss function
    device : torch.device
    epochs : int
    log_every : int
    figures_dir : Path or None
    log_fn : callable or None
        Called each epoch with a dict of metrics.
    cfg : dict or None
        Resolved config to embed in checkpoints.

    Returns
    -------
    dict
        Training results: best_val_loss, train_losses, val_losses, checkpoint_path.
    """
    from ML import plotting  # deferred to avoid circular imports

    best_val_loss = float("inf")
    train_losses = []
    val_losses = []
    checkpoint_path = None

    for epoch in range(1, epochs + 1):
        t_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        v_loss = eval_epoch(model, val_loader, criterion, device)
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        train_losses.append(t_loss)
        val_losses.append(v_loss)

        # Log metrics via callback
        if log_fn is not None:
            log_fn(
                {
                    "epoch": epoch,
                    "train_loss": t_loss,
                    "val_loss": v_loss,
                    "learning_rate": lr,
                }
            )

        if epoch % log_every == 0 or epoch == 1:
            print(
                f"Epoch {epoch:4d}/{epochs} | "
                f"Train: {t_loss:.6f} | Val: {v_loss:.6f} | LR: {lr:.2e}"
            )

        # Save best model
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            checkpoint_path = save_checkpoint(
                model,
                optimizer,
                epoch,
                v_loss,
                Path("checkpoints") / "best_model.pt",
                cfg=cfg,
            )

    # Save final model
    save_checkpoint(
        model,
        optimizer,
        epochs,
        val_losses[-1],
        Path("checkpoints") / "final_model.pt",
        cfg=cfg,
    )

    # Training plots
    if figures_dir is not None:
        figures_dir = Path(figures_dir)
        figures_dir.mkdir(parents=True, exist_ok=True)
        plotting.plot_loss_curves(train_losses, val_losses, figures_dir)

    print(f"Training complete. Best val loss: {best_val_loss:.6f}")

    return {
        "best_val_loss": best_val_loss,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "checkpoint_path": checkpoint_path,
    }
