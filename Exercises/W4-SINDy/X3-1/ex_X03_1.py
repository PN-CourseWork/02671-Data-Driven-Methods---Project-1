# Week 4 - Exercise X03.1
# DMD analysis of cylinder flow with Gavish-Donoho optimal rank selection
#
# (I)   Visualise the first few DMD modes
# (II)  Flowmap prediction to time step 200
# (III) Error growth for different truncation ranks r

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

plt.style.use("./science.mplstyle")

FIGURES_DIR = Path("figures/W4-SINDy/ex_X03_1")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Load data
# =============================================================================

mat = sio.loadmat("DATA/FLUIDS/CYLINDER_ALL.mat")
X = mat["VORTALL"].copy()
print(f"Data shape: {X.shape}")

X_mean = np.mean(X, axis=1, keepdims=True)
X_fluct = X - X_mean

# =============================================================================
# Gavish-Donoho optimal hard threshold
# =============================================================================


def omega(beta):
    """Optimal hard threshold coefficient (Gavish & Donoho 2014, Eq. 11)."""
    return np.sqrt(2 * (beta + 1) + (8 * beta) / (beta + 1 + np.sqrt(beta**2 + 14 * beta + 1)))


def gavish_donoho_rank(X):
    """Compute optimal SVD truncation rank via Gavish-Donoho threshold."""
    m, n = X.shape
    beta = min(m, n) / max(m, n)
    s = np.linalg.svd(X, compute_uv=False)
    tau = omega(beta) * np.median(s)
    r = np.sum(s > tau)
    return r, tau, s


r_opt, tau, s = gavish_donoho_rank(X)
print(f"beta = {min(X.shape) / max(X.shape):.6f}")
print(f"Optimal rank: {r_opt}, threshold: {tau:.6f}")

# =============================================================================
# DMD
# =============================================================================


def dmd(X, Xprime, r):
    """
    Exact DMD with rank-r truncation.

    Parameters
    ----------
    X : ndarray, shape (n, m)
        Snapshot matrix [x_0, ..., x_{m-1}].
    Xprime : ndarray, shape (n, m)
        Time-shifted snapshots [x_1, ..., x_m].
    r : int
        Truncation rank.

    Returns
    -------
    Phi : ndarray, shape (n, r)
        DMD modes.
    Lambda : ndarray, shape (r,)
        DMD eigenvalues.
    b : ndarray, shape (r,)
        Initial amplitudes.
    """
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    V = Vt.conj().T

    Ur = U[:, :r]
    Sr = np.diag(S[:r])
    Vr = V[:, :r]
    Sr_inv = np.diag(1.0 / S[:r])

    # Reduced operator
    Atilde = Ur.conj().T @ Xprime @ Vr @ Sr_inv

    # Eigen-decomposition
    Lambda_vals, W = np.linalg.eig(Atilde)

    # Exact DMD modes
    Phi = Xprime @ Vr @ Sr_inv @ W

    # Initial amplitudes (Eq. 7.31 in Brunton & Kutz)
    alpha1 = Sr @ Vr.conj().T[:, 0]
    b = np.linalg.lstsq(W @ np.diag(Lambda_vals), alpha1, rcond=None)[0]

    return Phi, Lambda_vals, b


# Build snapshot matrices
X1 = X_fluct[:, :-1]
X2 = X_fluct[:, 1:]

Phi, Lambda, b = dmd(X1, X2, r_opt)

# Reconstruction error
n_steps = X1.shape[1]
X_dmd = np.zeros((X.shape[0], n_steps), dtype=complex)
for k in range(n_steps):
    X_dmd[:, k] = Phi @ (b * (Lambda**k))

error = np.linalg.norm(X1 - X_dmd.real) / np.linalg.norm(X1)
print(f"Relative reconstruction error (r={r_opt}): {error:.6f}")

# =============================================================================
# (I) Visualise the first few DMD modes
# =============================================================================

nx, ny = 449, 199
num_modes = 6

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for i, ax in enumerate(axes.flat):
    if i >= num_modes:
        break
    ax.contourf(Phi[:, i].real.reshape(nx, ny), levels=50, cmap="jet")
    ax.set_title(f"DMD Mode {i}")
    ax.set_aspect("equal")

plt.tight_layout()
plt.savefig(FIGURES_DIR / "dmd_modes.pdf", bbox_inches="tight")
plt.close()
print(f"Saved DMD modes to {FIGURES_DIR}")

# =============================================================================
# (II) Flowmap prediction to time step 200
# =============================================================================

n_predict = 200
X_pred = np.zeros((X1.shape[0], n_predict), dtype=complex)
for k in range(n_predict):
    X_pred[:, k] = Phi @ (b * (Lambda**k))

X_pred_real = X_pred.real + X_mean

# Compare at a few time steps
compare_steps = [0, 50, 100, 150]
fig, axes = plt.subplots(len(compare_steps), 2, figsize=(10, 4 * len(compare_steps)))
for row, k in enumerate(compare_steps):
    axes[row, 0].contourf(X[:, k].reshape(nx, ny), levels=50, cmap="jet")
    axes[row, 0].set_title(f"True (t={k})")
    axes[row, 0].set_aspect("equal")
    axes[row, 1].contourf(X_pred_real[:, k].reshape(nx, ny), levels=50, cmap="jet")
    axes[row, 1].set_title(f"DMD (t={k})")
    axes[row, 1].set_aspect("equal")

plt.tight_layout()
plt.savefig(FIGURES_DIR / "flowmap_prediction.pdf", bbox_inches="tight")
plt.close()
print("Saved flowmap prediction comparison")

# =============================================================================
# (III) Error growth for different truncation ranks
# =============================================================================

r_values = [5, 10, 20, 40, 60]
n_error_steps = 150

plt.figure()
for r in r_values:
    Phi_r, Lambda_r, b_r = dmd(X1, X2, r)
    errors = []
    for k in range(n_error_steps):
        x_k = Phi_r @ (b_r * (Lambda_r**k))
        true = X_fluct[:, k]
        err = np.linalg.norm(true - x_k.real) / np.linalg.norm(true)
        errors.append(err)
    plt.plot(errors, label=f"$r={r}$")

plt.xlabel("Time step")
plt.ylabel("Relative error")
plt.title("DMD error growth for different truncation ranks")
plt.legend()
plt.grid(True)
plt.savefig(FIGURES_DIR / "error_growth_vs_rank.pdf", bbox_inches="tight")
plt.close()
print("Saved error growth plot")
