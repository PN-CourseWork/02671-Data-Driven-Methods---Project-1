"""Neural network model definitions for PDE time-stepping."""

import torch
import torch.nn as nn

ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
}


def _build_mlp(input_dim, output_dim, hidden_dims, activation="relu"):
    """Build a Sequential MLP with the given layer sizes."""
    act_fn = ACTIVATIONS[activation]
    layers = []
    in_dim = input_dim
    for h_dim in hidden_dims:
        layers += [nn.Linear(in_dim, h_dim), act_fn()]
        in_dim = h_dim
    layers.append(nn.Linear(in_dim, output_dim))
    return nn.Sequential(*layers)


class MLPTimestepper(nn.Module):
    """
    Generic MLP time-stepper: x(t) -> x(t + dt).

    Works for any input/output dimension — used directly for KS (1024-dim
    physical space) or SVD forecasting (r-dim coefficient space).

    Parameters
    ----------
    input_dim : int
        Input (and output) dimension.
    hidden_dims : list[int]
        Hidden layer sizes.
    activation : str
        One of "relu", "gelu", "tanh".
    """

    def __init__(self, input_dim, hidden_dims, activation="relu"):
        super().__init__()
        self.net = _build_mlp(input_dim, input_dim, list(hidden_dims), activation)

    def forward(self, x):
        return self.net(x)

    @torch.no_grad()
    def rollout(self, x0, n_steps):
        """Autoregressively advance from x0 for n_steps."""
        if x0.dim() == 1:
            x0 = x0.unsqueeze(0)
        trajectory = [x0]
        x = x0
        for _ in range(n_steps):
            x = self.forward(x)
            trajectory.append(x)
        return torch.cat(trajectory, dim=0)
