"""
Week 1 - Exercise X01.1
Setup a code that performs physics-informed least-squares regression to
determine the model parameters of a spread of disease model. Reproduce first the
results in the former slide example and then apply to a new spread of virus model. The
data to be used for testing can be artificially constructed using a numerical solver and
with known/own choice of parameters. For inspiration to select a model, cf. Shaier,
Raissi & Seshaiyer (2022).

"""

from pathlib import Path
import matplotlib.pyplot as plt

plt.style.use("./science.mplstyle")

FIGURES_DIR = Path("figures/W1-SVD/ex_1_7")


# =============================================================================
# 1.7(a) - Load data and perform SVD
# =============================================================================
