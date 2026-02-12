"""
Week 1 - Exercise X01.2
Setup a code that performs physics-informed least-squares regression to
determine a model parameter of a PDE such as the vorticity equation. Reproduce the
results of Nielsen et al. 2025 where the Reynolds number is estimated for Fluid flow
past a cylinder.
"""

from pathlib import Path
import matplotlib.pyplot as plt

plt.style.use("./science.mplstyle")

FIGURES_DIR = Path("figures/W1-SVD/ex_1_7")
