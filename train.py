"""
Single Hydra entry point for all ML training in 02671 Data-Driven Methods.

Usage:
    uv run python train.py +experiment=ex6_1a_ks
    uv run python train.py +experiment=ex6_1a_ks training=fast
    uv run python train.py +experiment=ex6_1a_ks training.learning_rate=5e-4
    uv run python train.py --multirun +experiment=ex6_1a_ks model=ks_mlp,ks_rnn
"""

from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

import wandb
from ML import trainer


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # -------------------------------------------------------------------------
    # Setup
    # -------------------------------------------------------------------------
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = trainer.get_device()
    model = hydra.utils.instantiate(cfg.model).to(device)
    dataset = hydra.utils.instantiate(cfg.data)

    model_name = cfg.model._target_.split(".")[-1]
    n_params = trainer.count_parameters(model)
    group = cfg.wandb.group or "default"
    figures_dir = Path("figures") / group

    print(f"Device: {device} | Model: {model_name} ({n_params:,} params)")
    print(f"Dataset: {len(dataset)} samples, dt = {dataset.dt:.4f}")

    # -------------------------------------------------------------------------
    # Resolve config to plain dict (for checkpoint embedding + wandb)
    # -------------------------------------------------------------------------
    resolved = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    # -------------------------------------------------------------------------
    # DataLoaders, optimizer, scheduler, loss
    # -------------------------------------------------------------------------
    train_loader, val_loader = trainer.make_dataloaders(
        dataset,
        train_frac=cfg.training.train_frac,
        batch_size=cfg.training.batch_size,
        seed=cfg.seed,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=cfg.training.scheduler_step,
        gamma=cfg.training.scheduler_gamma,
    )
    criterion = nn.MSELoss()

    # -------------------------------------------------------------------------
    # Wandb — init from Hydra config, context manager ensures run.finish()
    # -------------------------------------------------------------------------
    wandb_cfg = resolved["wandb"]

    with wandb.init(
        entity=wandb_cfg["entity"],
        project=wandb_cfg["project"],
        group=wandb_cfg["group"],
        job_type=wandb_cfg["job_type"],
        mode=wandb_cfg["mode"],
        name=f"{model_name}-lr{cfg.training.learning_rate}",
        tags=wandb_cfg["tags"] + [model_name],
        config=resolved,
    ) as run:
        # Track gradients and parameter distributions
        run.watch(model, log="gradients", log_freq=50)

        # ---------------------------------------------------------------------
        # Train
        # ---------------------------------------------------------------------
        results = trainer.fit(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=device,
            epochs=cfg.training.epochs,
            log_every=cfg.training.log_every,
            figures_dir=figures_dir,
            log_fn=run.log,
            cfg=resolved,
        )
        run.summary["best_val_loss"] = results["best_val_loss"]

        # ---------------------------------------------------------------------
        # Upload model artifact
        # ---------------------------------------------------------------------
        artifact_name = f"{group}-{model_name}"
        artifact = wandb.Artifact(
            name=artifact_name,
            type="model",
            description=f"Trained {model_name} for {group}",
            metadata={"best_val_loss": results["best_val_loss"], "model": model_name},
        )
        artifact.add_file(str(results["checkpoint_path"]))
        run.log_artifact(artifact)

        run.unwatch(model)

    print(f"Done. Artifact: {artifact_name}")


if __name__ == "__main__":
    main()
