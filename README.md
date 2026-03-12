# 02671 Data-Driven Methods — Project 1

## Quick Start

All commands run from **this directory** (`Project-1-Codebase/`).

```bash
uv sync                                              # Install dependencies
uv run python Exercises/W2-NN/ks_data_gen.py          # Generate data
uv run python train.py +experiment=ex6_1a_ks          # Train a model
uv run python Exercises/W1-SVD/ex1_7.py               # Run an analysis script
```

## Installation

```bash
uv sync
```

This installs all dependencies and the `ML` package (from `src/ML/`) in editable mode.

## Project Structure

```
train.py                # Hydra entry point for all ML training
conf/                   # Hydra configs
  config.yaml           #   Base defaults (model, data, training, wandb)
  model/                #   ks_mlp.yaml, svd_mlp.yaml
  data/                 #   ks.yaml
  training/             #   default.yaml (200 epochs), fast.yaml (10 epochs)
  experiment/           #   ex6_1a_ks.yaml (per-exercise overrides)
Exercises/              # Data generation, analysis, and evaluation scripts
  W1-SVD/               #   SVD, POD, DMD exercises
  W2-NN/                #   Neural network exercises
src/ML/                 # Shared library (models, dataset, trainer, evaluation, plotting)
figures/                # Generated PDF figures (organized by exercise)
DATA/                   # Generated data files (git-ignored)
science.mplstyle        # Matplotlib style (LaTeX, serif, tight layout)
```

## Usage

### 1. Generate data

Data generation scripts are standalone and run from the project root:

```bash
uv run python Exercises/W2-NN/ks_data_gen.py        # Kuramoto-Sivashinsky snapshots
uv run python Exercises/W1-SVD/ex_X01_1_DATA_GEN.py  # SIR model data
```

Output goes to `DATA/` (git-ignored).

### 2. Run exercise scripts

Exercise scripts run analysis, produce figures, and print diagnostics:

```bash
uv run python Exercises/W1-SVD/ex1_7.py              # SVD/POD/DMD of cylinder flow
```

Figures are saved as PDFs under `figures/`.

### 3. Train a model

All ML training goes through `train.py` with Hydra configs. An experiment override
(`+experiment=...`) is required to supply the wandb group and tags:

```bash
# Run an experiment
uv run python train.py +experiment=ex6_1a_ks

# Quick debug run (10 epochs)
uv run python train.py +experiment=ex6_1a_ks training=fast

# Override parameters from the CLI
uv run python train.py +experiment=ex6_1a_ks training.learning_rate=5e-4 training.epochs=100

# Sweep over model configs
uv run python train.py --multirun +experiment=ex6_1a_ks model=ks_mlp,svd_mlp

# Train offline (no wandb sync)
uv run python train.py +experiment=ex6_1a_ks wandb.mode=offline
```

Device selection is automatic: **CUDA > MPS (Apple Silicon) > CPU**.

### 4. Experiment tracking (wandb)

Runs log to wandb project `02671-DDMethods`, organized by:

- **Group** — exercise name (e.g., `ex6-1a`), set in the experiment YAML
- **Tags** — model type, dataset, exercise (auto-generated + configurable)
- **Artifacts** — model checkpoints are versioned and uploaded automatically

Retrieve a trained model artifact in evaluation scripts:

```python
artifact = wandb.use_artifact("ex6-1a-KSTimestepper:latest")
model_dir = artifact.download()
```

## Linting & Formatting

Ruff is configured in `pyproject.toml`:

```bash
uv run ruff check .            # Lint
uv run ruff check --fix .      # Lint + auto-fix
uv run ruff format .           # Format
uv run ruff format --check .   # Check formatting only
```

## Adding a New Exercise

1. **Data generation** — add a standalone script in `Exercises/W<N>-<topic>/`
2. **Model** — add a class in `src/ML/models.py` and a YAML in `conf/model/`
3. **Dataset** — add a class in `src/ML/dataset.py` and a YAML in `conf/data/`
4. **Experiment** — create `conf/experiment/<name>.yaml` with the right overrides
5. **Train** — `uv run python train.py +experiment=<name>`
