"""Utility functions for neural structural optimization models."""

import numpy as np
import torch


def batched_topo_loss(params, envs):
    """Compute batched topology loss for multiple environments."""
    losses = [env.objective(params[i], volume_constraint=True)
              for i, env in enumerate(envs)]
    return np.stack(losses)


def set_random_seed(seed):
    """Set random seed for both NumPy and PyTorch."""
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
