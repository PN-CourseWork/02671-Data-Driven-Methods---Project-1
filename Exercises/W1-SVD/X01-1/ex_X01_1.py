# Week 1 - Exercise X01.1
# Physics-informed least-squares regression for SIR model parameters
#
# Generate synthetic SIR data, estimate derivatives via finite differences,
# then recover beta and gamma via least-squares regression.

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

plt.style.use("./science.mplstyle")

FIGURES_DIR = Path("figures/W1-SVD/ex_X01_1")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Generate synthetic SIR data
# =============================================================================

beta_true = 0.8
gamma_true = 1.0 / 5.4  # ~0.1852


def sir_model(t, y, beta, gamma):
    S, Inf, R = y
    dSdt = -beta * S * Inf
    dIdt = beta * S * Inf - gamma * Inf
    dRdt = gamma * Inf
    return [dSdt, dIdt, dRdt]


y0 = [1 - 0.001, 0.001, 0.0]
tspan = (0.0, 30.0)
dt = 0.1
t_eval = np.arange(tspan[0], tspan[1] + dt, dt)

sol = solve_ivp(lambda t, y: sir_model(t, y, beta_true, gamma_true), tspan, y0, t_eval=t_eval)
S, Inf, R = sol.y
t = sol.t

# Plot synthetic data
plt.figure(figsize=(8, 4))
plt.plot(t, S, "b", label="S")
plt.plot(t, Inf, "r", label="I")
plt.plot(t, R, "g", label="R")
plt.xlabel("Time [days]")
plt.ylabel("Population fraction")
plt.title("Synthetic SIR Data")
plt.legend()
plt.savefig(FIGURES_DIR / "sir_data.pdf", bbox_inches="tight")
plt.close()

# =============================================================================
# Finite difference derivatives
# =============================================================================


def finite_difference(u, dt):
    """Central differences (interior), forward/backward at boundaries."""
    dudt = np.zeros_like(u)
    dudt[1:-1] = (u[2:] - u[:-2]) / (2 * dt)  # central
    dudt[0] = (u[1] - u[0]) / dt  # forward
    dudt[-1] = (u[-1] - u[-2]) / dt  # backward
    return dudt


dSdt = finite_difference(S, dt)
dIdt = finite_difference(Inf, dt)

# =============================================================================
# Build regression system and solve
# =============================================================================

# Use interior points (skip boundary finite-difference artifacts)
idx = slice(1, -1)
S_i = S[idx]
I_i = Inf[idx]
dS_i = dSdt[idx]
dI_i = dIdt[idx]

N = len(S_i)

# System: [dS/dt; dI/dt] = A * [beta; gamma]
#   dS/dt = -beta * S * I           ->  row: [-S*I,  0   ] * [beta; gamma]
#   dI/dt =  beta * S * I - gamma*I ->  row: [ S*I, -I   ] * [beta; gamma]
A = np.zeros((2 * N, 2))
b = np.zeros(2 * N)

A[0:N, 0] = -S_i * I_i
b[0:N] = dS_i

A[N : 2 * N, 0] = S_i * I_i
A[N : 2 * N, 1] = -I_i
b[N : 2 * N] = dI_i

theta_hat, *_ = np.linalg.lstsq(A, b, rcond=None)
beta_hat = theta_hat[0]
gamma_hat = theta_hat[1]

print(f"True beta:       {beta_true}")
print(f"Estimated beta:  {beta_hat:.6f}")
print(f"True gamma:      {gamma_true:.6f}")
print(f"Estimated gamma: {gamma_hat:.6f}")
print(f"Beta  error:     {abs(beta_hat - beta_true) / beta_true:.6e}")
print(f"Gamma error:     {abs(gamma_hat - gamma_true) / gamma_true:.6e}")
