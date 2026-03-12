# Week 4 - Exercise 7.4
# SINDy (Sparse Identification of Nonlinear Dynamics) on the Lorenz system
#
# (a) Minimum data length T for correct identification
# (b) Effect of sampling rate dt on identification
# (c) Noise sensitivity analysis

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

plt.style.use("./science.mplstyle")

FIGURES_DIR = Path("figures/W4-SINDy/ex7_4")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Lorenz system
# =============================================================================

SIGMA = 10
RHO = 28
BETA = 8 / 3


def lorenz(t, x):
    return [
        SIGMA * (x[1] - x[0]),
        x[0] * (RHO - x[2]) - x[1],
        x[0] * x[1] - BETA * x[2],
    ]


# =============================================================================
# SINDy building blocks
# =============================================================================


def polynomial_library(X):
    """Build polynomial feature library up to degree 2."""
    x, y, z = X[:, 0], X[:, 1], X[:, 2]
    return np.column_stack(
        [
            np.ones(len(x)),
            x,
            y,
            z,
            x * y,
            x * z,
            y * z,
            x**2,
            y**2,
            z**2,
        ]
    )


def sindy(Theta, dX, lam=0.01, n_iter=5):
    """Sparse regression via sequential thresholded least squares (STLS)."""
    Xi = np.linalg.lstsq(Theta, dX, rcond=None)[0]

    for _ in range(n_iter):
        small = np.abs(Xi) < lam
        Xi[small] = 0

        for i in range(dX.shape[1]):
            big_idx = Xi[:, i] != 0
            if np.sum(big_idx) == 0:
                continue
            Xi[big_idx, i] = np.linalg.lstsq(Theta[:, big_idx], dX[:, i], rcond=None)[0]

    return Xi


def check_lorenz_structure(Xi, tol=1e-3):
    """Check if identified coefficient matrix has the correct Lorenz sparsity."""
    structure = np.abs(Xi) > tol
    return (
        structure[1, 0]
        and structure[2, 0]  # dx: x, y
        and structure[1, 1]
        and structure[3, 1]
        and structure[5, 1]  # dy: x, z, xz
        and structure[4, 2]
        and structure[3, 2]  # dz: xy, z
    )


# =============================================================================
# 7.4(a) — Minimum data length for identification
# =============================================================================

print("=" * 60)
print("7.4(a) — Sweep data length T")
print("=" * 60)

dt = 0.0001
t_final = 10
t_eval = np.arange(0, t_final, dt)
sol = solve_ivp(lorenz, [0, t_final], [1, 1, 1], t_eval=t_eval)

t = sol.t
X_data = sol.y.T

# Discard transient (first t=1)
mask = t > 1
t = t[mask]
X_data = X_data[mask]

# Clean derivatives
dX_data = np.array([lorenz(t_i, x_i) for t_i, x_i in zip(t, X_data, strict=True)])

T_values = np.linspace(0.1, 4, 30)
correct_T = None

for T in T_values:
    n = int(T / dt)
    if n >= len(X_data):
        break

    Theta = polynomial_library(X_data[:n])
    Xi = sindy(Theta, dX_data[:n], lam=0.01)

    if check_lorenz_structure(Xi):
        correct_T = T
        print(f"\nIdentified at T = {T:.4f}")
        print(np.round(Xi, 3))
        break

if correct_T is None:
    print("Model not identified in sweep range")
else:
    # Plot the trajectory segment used
    n_used = int(correct_T / dt)
    plt.figure(figsize=(8, 6))
    plt.plot(X_data[:n_used, 0], X_data[:n_used, 1])
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.title(f"Lorenz trajectory used for identification ($T = {correct_T:.2f}$)")
    plt.savefig(FIGURES_DIR / "min_data_trajectory.pdf", bbox_inches="tight")
    plt.close()

print(f"Minimum T for correct model: {correct_T}")

# =============================================================================
# 7.4(b) — Effect of sampling rate on identification
# =============================================================================

print("\n" + "=" * 60)
print("7.4(b) — Sweep sampling rate dt")
print("=" * 60)

dt_values = np.logspace(-4, -1, 8)
results = []

for dt_test in dt_values:
    t_span = (0, 5)
    t_eval_test = np.arange(0, 5, dt_test)
    sol_test = solve_ivp(lorenz, t_span, [-8, 8, 27], t_eval=t_eval_test)

    X_test = sol_test.y.T
    dX_test = np.array([lorenz(0, x) for x in X_test])

    # Discard transient
    discard = int(1 / dt_test)
    X_test = X_test[discard:]
    dX_test = dX_test[discard:]

    T_sweep = np.linspace(0.1, 3, 20)
    found_T = None

    for T in T_sweep:
        n = int(T / dt_test)
        if n >= len(X_test):
            break

        Theta = polynomial_library(X_test[:n])
        Xi = sindy(Theta, dX_test[:n], lam=0.05)

        if check_lorenz_structure(Xi):
            found_T = T
            break

    N_samples = int(found_T / dt_test) if found_T is not None else None
    results.append((dt_test, found_T, N_samples))
    status = f"T={found_T:.3f}, N={N_samples}" if found_T else "not identified"
    print(f"  dt={dt_test:.5f} | {status}")

# Plot results
identified = [(dt_v, T, N) for dt_v, T, N in results if T is not None]
if identified:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    dts = [r[0] for r in identified]
    Ts = [r[1] for r in identified]
    Ns = [r[2] for r in identified]

    ax1.semilogx(dts, Ts, "o-")
    ax1.set_xlabel("$\\Delta t$")
    ax1.set_ylabel("Minimum $T$")
    ax1.set_title("Required data length vs sampling rate")
    ax1.grid(True)

    ax2.semilogx(dts, Ns, "o-")
    ax2.set_xlabel("$\\Delta t$")
    ax2.set_ylabel("Minimum $N$ samples")
    ax2.set_title("Required samples vs sampling rate")
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "sampling_rate_sweep.pdf", bbox_inches="tight")
    plt.close()

# =============================================================================
# 7.4(c) — Noise sensitivity
# =============================================================================

print("\n" + "=" * 60)
print("7.4(c) — Noise sensitivity")
print("=" * 60)

dt_noise = 0.01
t_span = (0, 5)
t_eval_noise = np.arange(0, 5, dt_noise)
sol_noise = solve_ivp(lorenz, t_span, [-8, 8, 27], t_eval=t_eval_noise)

X_clean = sol_noise.y.T
dX_clean = np.array([lorenz(0, x) for x in X_clean])

# Discard transient
discard = int(1 / dt_noise)
X_clean = X_clean[discard:]
dX_clean = dX_clean[discard:]

noise_levels = [0.0, 0.01, 0.05, 0.1]
num_realizations = 20
T_noise_values = np.linspace(0.1, 3, 20)

plt.figure(figsize=(8, 5))

for noise_std in noise_levels:
    success_rates = []

    for T in T_noise_values:
        n = int(T / dt_noise)
        if n >= len(X_clean):
            break

        success_count = 0
        for _ in range(num_realizations):
            noise = noise_std * np.random.randn(*X_clean[:n].shape)
            X_noisy = X_clean[:n] + noise

            Theta = polynomial_library(X_noisy)
            Xi = sindy(Theta, dX_clean[:n], lam=0.1)

            if check_lorenz_structure(Xi):
                success_count += 1

        success_rates.append(success_count / num_realizations)

    plt.plot(T_noise_values[: len(success_rates)], success_rates, label=f"$\\sigma = {noise_std}$")
    print(f"  noise={noise_std:.2f} | success rates: {[f'{s:.1f}' for s in success_rates[:5]]}...")

plt.xlabel("Data length $T$")
plt.ylabel("Success rate")
plt.title("SINDy identification success rate vs data length with noise")
plt.legend()
plt.grid(True)
plt.savefig(FIGURES_DIR / "noise_sensitivity.pdf", bbox_inches="tight")
plt.close()
print(f"\nAll figures saved to {FIGURES_DIR}")
