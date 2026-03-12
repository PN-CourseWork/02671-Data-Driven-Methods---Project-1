# Week 1 - Exercise 1.7
# SVD Analysis of Cylinder Flow Data

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
plt.style.use("science.mplstyle")

FIGURES_DIR = Path("figures/W1-SVD/ex_1_7")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

### LOAD DATA
file_path = Path("DATA") / "FLUIDS" / "ibpmUNSTEADY.plt"
data = np.loadtxt(file_path, skiprows=6)

mat = sio.loadmat("DATA/FLUIDS/CYLINDER_ALL.mat")

VORTALL = mat["VORTALL"]
print(VORTALL.shape)
X = VORTALL.copy()

X_mean = np.mean(X, axis=1, keepdims=True)
X_fluct = X - X_mean

### SVD decomposition
U, S, Vt = np.linalg.svd(X_fluct, full_matrices=False)
# U spatial modes
# S singular values
# Vt time dynamics

# Plot singular value spectrum
plt.figure()
plt.semilogy(S / S[0], "o-")
plt.xlabel("Mode index")
plt.ylabel("Normalized singular value")
plt.title("Singular Value Spectrum")
plt.grid(True)
plt.savefig(FIGURES_DIR / "singular_value_spectrum.pdf", bbox_inches="tight")
plt.close()

# Plot the leading singular vectors (U matrix)
nx, ny = 449, 199

mode1 = U[:, 0].reshape(nx, ny)
mode2 = U[:, 1].reshape(nx, ny)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
c1 = axes[0].contourf(mode1, levels=50)
fig.colorbar(c1, ax=axes[0])
axes[0].set_title("POD Mode 1 (Eigenflow)")
c2 = axes[1].contourf(mode2, levels=50)
fig.colorbar(c2, ax=axes[1])
axes[1].set_title("POD Mode 2 (Eigenflow)")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "pod_modes.pdf", bbox_inches="tight")
plt.close()

# Plot ΣV* (temporal amplitudes)
time_coeffs = np.diag(S) @ Vt

fig, axes = plt.subplots(1, 2, figsize=(10, 3))
axes[0].plot(time_coeffs[0, :])
axes[0].set_xlabel("Time index")
axes[0].set_ylabel("Amplitude")
axes[0].set_title("Temporal Coefficient of Mode 1")
axes[1].plot(time_coeffs[1, :])
axes[1].set_xlabel("Time index")
axes[1].set_ylabel("Amplitude")
axes[1].set_title("Temporal Coefficient of Mode 2")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "temporal_coefficients.pdf", bbox_inches="tight")
plt.close()

# =============================================================================
# 1.7(b) - Low-rank reconstruction
# =============================================================================

# Compute truncation values r
energy = S**2
energy_fraction = np.cumsum(energy) / np.sum(energy)

r_90 = np.searchsorted(energy_fraction, 0.90) + 1
r_99 = np.searchsorted(energy_fraction, 0.99) + 1
r_999 = np.searchsorted(energy_fraction, 0.999) + 1

print(r_90, r_99, r_999)


# Low-rank reconstruction
def reconstruct(U, S, Vt, r):
    Ur = U[:, :r]
    Sr = np.diag(S[:r])
    Vtr = Vt[:r, :]
    return Ur @ Sr @ Vtr


# Reconstruct the flow movies
Xr_90 = reconstruct(U, S, Vt, r_90)
Xr_99 = reconstruct(U, S, Vt, r_99)
Xr_999 = reconstruct(U, S, Vt, r_999)

Xr_90 += X_mean
Xr_99 += X_mean
Xr_999 += X_mean

# plot movies
nx, ny = 449, 199
t = 50  # pick a representative time index

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.contourf(X[:, t].reshape(nx, ny), levels=50)
plt.title("True")

plt.subplot(1, 3, 2)
plt.contourf(Xr_90[:, t].reshape(nx, ny), levels=50)
plt.title(f"r = {r_90} (90%)")

plt.subplot(1, 3, 3)
plt.contourf(Xr_99[:, t].reshape(nx, ny), levels=50)
plt.title(f"r = {r_99} (99%)")

plt.tight_layout()
plt.savefig(FIGURES_DIR / "low_rank_reconstruction.pdf", bbox_inches="tight")
plt.close()


# Squared Frobenius norm of the error
def frob_error(X, Xr):
    return np.linalg.norm(X - Xr, "fro") ** 2


err_90 = frob_error(X, Xr_90)
err_99 = frob_error(X, Xr_99)
err_999 = frob_error(X, Xr_999)

print(err_90, err_99, err_999)

# =============================================================================
# 1.7(c) - POD amplitude matrix W
# =============================================================================

# Truncate for r=10
r = 10

U_tilde = U[:, :r]  # (89351, 10)
Sigma_tilde = np.diag(S[:r])  # (10, 10)
Vt_tilde = Vt[:r, :]  # (10, 151)

# Define matrix W = \tilde{\Sigma}\tilde{V}^*
W = Sigma_tilde @ Vt_tilde

# Comparison
k = 50  # any snapshot index
w_k = W[:, k]
x_k = X_fluct[:, k]  # mean-subtracted snapshot
x_k_recon = U_tilde @ W[:, k]

# Comparison plot
nx, ny = 449, 199

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.contourf(x_k.reshape(nx, ny), levels=50)
plt.colorbar()
plt.title("True snapshot")

plt.subplot(1, 2, 2)
plt.contourf(x_k_recon.reshape(nx, ny), levels=50)
plt.colorbar()
plt.title("Reconstructed: $\\tilde{U} w_k$")

plt.tight_layout()
plt.savefig(FIGURES_DIR / "pod_reconstruction_comparison.pdf", bbox_inches="tight")
plt.close()

# Quantatative comparison
error = np.linalg.norm(x_k - x_k_recon) / np.linalg.norm(x_k)
print("Relative error:", error)

# =============================================================================
# 1.7(d) - DMD matrix A
# =============================================================================

W1 = W[:, :-1]  # w_1, w_2, ..., w_{m-1}
W2 = W[:, 1:]  # w_2, w_3, ..., w_m
# The psuedoinverse of W
Uw, Sw, Vtw = np.linalg.svd(W1, full_matrices=False)

Sw_inv = np.diag(1 / Sw)
W1_pinv = Vtw.T @ Sw_inv @ Uw.T
# Compute Matrix A
A = W2 @ W1_pinv
print(A.shape)

# Compute eigenvals of A
eigvals, eigvecs = np.linalg.eig(A)
# Plot eigenvals in complex plane
plt.figure()
plt.scatter(eigvals.real, eigvals.imag)
plt.axhline(0, color="k")
plt.axvline(0, color="k")
plt.gca().set_aspect("equal", adjustable="box")
plt.xlabel("Real part")
plt.ylabel("Imaginary part")
plt.title("Eigenvalues of A")
plt.grid(True)
plt.savefig(FIGURES_DIR / "eigenvalues_A.pdf", bbox_inches="tight")
plt.close()

# =============================================================================
# 1.7(e) - Time advancement and flow prediction
# =============================================================================

w1 = W[:, 0]  # initial POD amplitudes
r, m = W.shape

W_pred = np.zeros((r, m))
W_pred[:, 0] = w1

# Advance in time
for k in range(1, m):
    W_pred[:, k] = A @ W_pred[:, k - 1]

# Reconstruct the flow from predicted amplitudes
X_pred = U_tilde @ W_pred
X_pred += X_mean

# Comparison of predicted vs true flow fields
k = 50  # any snapshot index
nx, ny = 449, 199

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.contourf(X[:, k].reshape(nx, ny), levels=50)
plt.colorbar()
plt.title("True flow field")

plt.subplot(1, 2, 2)
plt.contourf(X_pred[:, k].reshape(nx, ny), levels=50)
plt.colorbar()
plt.title("Predicted flow field")

plt.tight_layout()
plt.savefig(FIGURES_DIR / "flow_prediction_comparison.pdf", bbox_inches="tight")
plt.close()

# quantify prediction error
rel_error = np.linalg.norm(X[:, k] - X_pred[:, k]) / np.linalg.norm(X[:, k])
print("Relative error:", rel_error)
