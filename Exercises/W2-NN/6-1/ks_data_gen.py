"""
Kuramoto-Sivashinsky equation solver (ETDRK4 spectral method).

Ported from MATLAB code (Kassam-Trefethen ETDRK4 scheme).
    u_t = -u*u_x - u_xx - u_xxxx,  periodic BCs

Generates training data for neural network time-stepping.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.style.use("./science.mplstyle")

FIGURES_DIR = Path("figures/W2-NN/ex_6_1a")
DATA_DIR = Path("DATA/W2-NN/KS")


def ks_etdrk4(u0, x, h, tmax, nplt=1):
    """
    Solve the Kuramoto-Sivashinsky equation using the ETDRK4 scheme.

    Parameters
    ----------
    u0 : ndarray, shape (N,)
        Initial condition in physical space.
    x : ndarray, shape (N,)
        Spatial grid points.
    h : float
        Time step size.
    tmax : float
        Final time.
    nplt : int
        Save every nplt-th time step.

    Returns
    -------
    uu : ndarray, shape (N, n_saved)
        Solution snapshots.
    tt : ndarray, shape (n_saved,)
        Corresponding time values.
    """
    N = len(u0)
    v = np.fft.fft(u0)

    # Wavenumbers (matching MATLAB convention)
    k = (
        np.concatenate(
            [
                np.arange(0, N // 2),
                [0],
                np.arange(-N // 2 + 1, 0),
            ]
        )
        / 16.0
    )

    # Linear operator in Fourier space: L = k^2 - k^4
    L = k**2 - k**4

    # ETDRK4 coefficients (Kassam-Trefethen)
    E = np.exp(h * L)
    E2 = np.exp(h * L / 2)

    # Contour integral evaluation of phi-functions
    M = 16
    r = np.exp(1j * np.pi * (np.arange(1, M + 1) - 0.5) / M)  # (M,)

    # LR shape: (N, M) — L broadcast with contour points
    LR = h * L[:, None] + r[None, :]  # (N, M)

    Q = h * np.real(np.mean((np.exp(LR / 2) - 1) / LR, axis=1))
    f1 = h * np.real(np.mean((-4 - LR + np.exp(LR) * (4 - 3 * LR + LR**2)) / LR**3, axis=1))
    f2 = h * np.real(np.mean((2 + LR + np.exp(LR) * (-2 + LR)) / LR**3, axis=1))
    f3 = h * np.real(np.mean((-4 - 3 * LR - LR**2 + np.exp(LR) * (4 - LR)) / LR**3, axis=1))

    # Nonlinear term prefactor: g = -0.5i * k
    g = -0.5j * k

    # Time-stepping loop
    nmax = round(tmax / h)
    uu_list = [u0.copy()]
    tt_list = [0.0]

    for n in range(1, nmax + 1):
        t = n * h
        # ETDRK4 stages
        Nv = g * np.fft.fft(np.real(np.fft.ifft(v)) ** 2)
        a = E2 * v + Q * Nv
        Na = g * np.fft.fft(np.real(np.fft.ifft(a)) ** 2)
        b = E2 * v + Q * Na
        Nb = g * np.fft.fft(np.real(np.fft.ifft(b)) ** 2)
        c = E2 * a + Q * (2 * Nb - Nv)
        Nc = g * np.fft.fft(np.real(np.fft.ifft(c)) ** 2)
        v = E * v + Nv * f1 + 2 * (Na + Nb) * f2 + Nc * f3

        if n % nplt == 0:
            u = np.real(np.fft.ifft(v))
            uu_list.append(u)
            tt_list.append(t)

    uu = np.column_stack(uu_list)  # (N, n_saved)
    tt = np.array(tt_list)
    return uu, tt


def generate_ks_data():
    """Generate KS training data with the default initial condition."""
    N = 1024
    x = 32 * np.pi * np.arange(1, N + 1) / N
    u0 = np.cos(x / 16) * (1 + np.sin(x / 16))

    h = 0.025
    tmax = 100.0
    # Save every 4th step -> dt_save = 0.1
    nplt = 4

    print("Solving KS equation...")
    uu, tt = ks_etdrk4(u0, x, h, tmax, nplt=nplt)
    print(f"Solution shape: {uu.shape}, time points: {len(tt)}")
    print(f"Time range: [{tt[0]:.2f}, {tt[-1]:.2f}], dt_save = {tt[1] - tt[0]:.4f}")

    # Save data
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.save(DATA_DIR / "ks_uu.npy", uu)
    np.save(DATA_DIR / "ks_tt.npy", tt)
    np.save(DATA_DIR / "ks_x.npy", x)
    print(f"Data saved to {DATA_DIR}/")

    return uu, tt, x


def plot_ks_solution(uu, tt, x):
    """Plot a pcolor visualization of the KS solution."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    T, X = np.meshgrid(tt, x)
    pcm = ax.pcolormesh(T, X, uu, shading="auto", cmap="hot")
    fig.colorbar(pcm, ax=ax)
    ax.set_xlabel("Time $t$")
    ax.set_ylabel("Space $x$")
    ax.set_title("Kuramoto--Sivashinsky equation")
    plt.savefig(FIGURES_DIR / "ks_solution.pdf", bbox_inches="tight")
    plt.close()
    print(f"Figure saved to {FIGURES_DIR / 'ks_solution.pdf'}")


if __name__ == "__main__":
    uu, tt, x = generate_ks_data()
    plot_ks_solution(uu, tt, x)
