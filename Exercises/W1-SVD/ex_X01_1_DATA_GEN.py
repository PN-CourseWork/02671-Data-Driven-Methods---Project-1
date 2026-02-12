import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from pathlib import Path


plt.style.use("./science.mplstyle")

FIGURES_DIR = Path("figures/W1-SVD/ex_X01_1")


# RHS function


def sir_model(t, y, beta, gamma):
    # Define the system of differential equations
    _S, _I, _R = y
    dSdt = -beta * _S * _I
    dIdt = beta * _S * _I - gamma * _I
    dRdt = gamma * _I
    return [dSdt, dIdt, dRdt]


# Define the initial conditions and parameter values
y0 = [1 - 0.001, 0.001, 0]  # [S, I, R]
beta = 0.8
gamma = 1.0 / 5.4
tspan = (0, 30)  # Define the time span for the simulation
# Solve the system of differential equations using solve_ivp
tmeasurements = np.linspace(0, 30, 100)
sol = solve_ivp(
    lambda t, y: sir_model(t, y, beta, gamma), tspan, y0, t_eval=tmeasurements
)

print(f"Shape of solution: {type(sol)}")

# Plot the results
plt.plot(sol.t, sol.y[0], "b", sol.t, sol.y[1], "r", sol.t, sol.y[2], "g")
plt.legend(["S", "I", "R"])
plt.savefig(FIGURES_DIR / "Num_Sim.pdf", bbox_inches="tight")
