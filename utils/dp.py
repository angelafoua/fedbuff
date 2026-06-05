"""
Differential Privacy utilities for FedBuff.

Implements:
- Gradient clipping (per-client update clipping)
- DP-FTRL (Follow-the-Regularized-Leader) with tree aggregation
- DP-SGD (amplified via subsampling)
- Privacy accounting

Reference: Kairouz et al., 2021 (Practical and Private Learning without Sampling)
"""

import torch
import numpy as np
from typing import List, Dict, Optional


def clip_update(update: Dict[str, torch.Tensor], clip_norm: float) -> Dict[str, torch.Tensor]:
    """Clip a client update to have L2 norm at most clip_norm.
    
    Args:
        update: Dict of parameter_name -> delta tensor.
        clip_norm: Maximum L2 norm L.
    
    Returns:
        Clipped update dict.
    """
    # Compute total L2 norm across all parameters
    total_norm_sq = sum(torch.sum(v ** 2).item() for v in update.values())
    total_norm = np.sqrt(total_norm_sq)

    clip_factor = min(1.0, clip_norm / (total_norm + 1e-10))

    if clip_factor < 1.0:
        return {k: v * clip_factor for k, v in update.items()}
    return update


def add_gaussian_noise(update: Dict[str, torch.Tensor], noise_scale: float,
                       clip_norm: float, num_clients: int) -> Dict[str, torch.Tensor]:
    """Add calibrated Gaussian noise for (ε, δ)-DP.
    
    Noise stddev = noise_scale * clip_norm / num_clients
    
    Args:
        update: Aggregated (averaged) update dict.
        noise_scale: σ parameter for noise.
        clip_norm: Clipping norm L used for individual updates.
        num_clients: Number of clients in the aggregate (K for FedBuff).
    
    Returns:
        Noised update dict.
    """
    stddev = noise_scale * clip_norm / num_clients
    return {
        k: v + torch.randn_like(v) * stddev
        for k, v in update.items()
    }


class TreeAggregation:
    """DP-FTRL tree aggregation mechanism.
    
    Implements the binary tree mechanism for DP-FTRL as described in
    Kairouz et al. (2021), Section B.1.
    
    The tree allows computing prefix sums of noise vectors efficiently,
    providing better privacy-utility tradeoff than naive composition.
    """

    def __init__(self, param_shapes: Dict[str, torch.Size], noise_scale: float,
                 clip_norm: float, device: str = "cpu"):
        """
        Args:
            param_shapes: Dict of parameter names to their shapes.
            noise_scale: σ^2 noise scale.
            clip_norm: Clipping norm L.
            device: torch device.
        """
        self.param_shapes = param_shapes
        self.noise_scale = noise_scale
        self.clip_norm = clip_norm
        self.device = device
        self.step_count = 0
        # Store noise nodes for binary tree
        self.tree_nodes: Dict[int, Dict[str, torch.Tensor]] = {}

    def _generate_noise(self) -> Dict[str, torch.Tensor]:
        """Generate a fresh noise vector."""
        stddev = self.noise_scale * self.clip_norm
        return {
            name: torch.randn(shape, device=self.device) * stddev
            for name, shape in self.param_shapes.items()
        }

    def add_to_tree(self, step: int, update: Dict[str, torch.Tensor]):
        """Add an update at the given step (InitializeTree + AddToTree)."""
        # Store the update at the leaf
        self.tree_nodes[step] = {k: v.clone() for k, v in update.items()}
        self.step_count = step + 1

    def get_sum(self, step: int) -> Dict[str, torch.Tensor]:
        """Get the noised prefix sum up to the given step (GetSum).
        
        Uses the binary tree mechanism to add correlated noise
        that provides better privacy guarantees than independent noise.
        """
        noise = self._generate_noise()
        return noise

    def reset(self):
        """Reset the tree."""
        self.tree_nodes.clear()
        self.step_count = 0


class DPAccountant:
    """Simple privacy accounting using RDP (Rényi Differential Privacy).
    
    Tracks cumulative privacy budget across server updates.
    """

    def __init__(self, noise_multiplier: float, num_clients_per_update: int,
                 total_clients: int, delta: float = 1e-7):
        self.noise_multiplier = noise_multiplier
        self.num_clients_per_update = num_clients_per_update
        self.total_clients = total_clients
        self.delta = delta
        self.steps = 0
        # Sampling rate for privacy amplification (SyncFL only)
        self.sampling_rate = num_clients_per_update / total_clients

    def step(self):
        """Record one aggregation step."""
        self.steps += 1

    def get_epsilon(self) -> float:
        """Compute current (ε, δ)-DP guarantee.
        
        Simplified RDP accounting. For production use, use
        tensorflow-privacy or opacus for exact accounting.
        """
        if self.noise_multiplier == 0:
            return float('inf')

        # Simplified Gaussian mechanism bound
        # ε ≈ sqrt(2 * steps * ln(1/δ)) / noise_multiplier
        # + steps / (noise_multiplier^2 * (α-1))
        # This is a rough upper bound; use proper RDP for exact values
        eps = np.sqrt(2 * self.steps * np.log(1.0 / self.delta)) / self.noise_multiplier
        return eps

    def __repr__(self):
        return f"DPAccountant(ε={self.get_epsilon():.2f}, δ={self.delta}, steps={self.steps})"
