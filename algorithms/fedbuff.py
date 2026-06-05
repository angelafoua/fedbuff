"""
FedBuff: Buffered Asynchronous Aggregation (Algorithm 1 from the paper).

Key idea: In asynchronous FL, instead of updating the server model on every
client update (FedAsync), buffer K updates before performing a server step.
This enables SecAgg compatibility and better scalability.

The buffer can run inside a TEE for privacy. K is independent of concurrency,
providing an extra degree of freedom over SyncFL.
"""

from typing import Dict, List, Optional
import torch
from algorithms.base import FLServer, ClientUpdate
from utils.delay import staleness_scaling
from utils.dp import clip_update, add_gaussian_noise, TreeAggregation


class FedBuffServer(FLServer):
    """FedBuff server with buffered asynchronous aggregation.
    
    Algorithm:
        1. Clients asynchronously download model, train locally, upload updates
        2. Server collects updates into a buffer
        3. When buffer has K updates, aggregate and perform server step
        4. Reset buffer and repeat
    """

    def __init__(self, model, server_lr: float = 1.0, momentum: float = 0.0,
                 buffer_size: int = 10, staleness_alpha: float = 0.5,
                 lr_norm: bool = True, device: str = "cpu",
                 # DP parameters
                 dp_enabled: bool = False, clip_norm: float = 1.0,
                 noise_scale: float = 0.0, dp_mode: str = "ftrl"):
        super().__init__(model, server_lr, momentum, device)
        self.buffer_size = buffer_size  # K
        self.staleness_alpha = staleness_alpha
        self.lr_norm = lr_norm

        # Buffer state
        self.buffer: List[ClientUpdate] = []

        # DP
        self.dp_enabled = dp_enabled
        self.clip_norm = clip_norm
        self.noise_scale = noise_scale
        self.dp_mode = dp_mode
        self.tree_agg: Optional[TreeAggregation] = None

        if dp_enabled and dp_mode == "ftrl":
            param_shapes = {name: p.shape for name, p in model.named_parameters()}
            self.tree_agg = TreeAggregation(param_shapes, noise_scale, clip_norm, device)

    def receive_update(self, update: ClientUpdate) -> bool:
        """Receive a single client update into the buffer.
        
        Returns True if this triggered a server update (buffer full).
        """
        # DP: clip the update inside TEE
        if self.dp_enabled:
            update.delta = clip_update(update.delta, self.clip_norm)

        self.buffer.append(update)
        self.total_client_trips += 1

        if len(self.buffer) >= self.buffer_size:
            self._aggregate_and_step()
            return True
        return False

    def process_updates(self, updates: List[ClientUpdate]) -> bool:
        """Process a batch of updates (for simulation convenience)."""
        updated = False
        for u in updates:
            if self.receive_update(u):
                updated = True
        return updated

    def _aggregate_and_step(self):
        """Aggregate K buffered updates and perform server step (Lines 10-15)."""
        K = len(self.buffer)

        # Aggregate: Δ_t = (1/K) * Σ s(τ_i) * Δ_i
        aggregated = {}
        total_weight = 0.0

        for update in self.buffer:
            weight = staleness_scaling(update.staleness, self.staleness_alpha)
            total_weight += weight

            for name, delta in update.delta.items():
                if name not in aggregated:
                    aggregated[name] = torch.zeros_like(delta)
                aggregated[name] += weight * delta

        # Normalize by K (or total weight)
        for name in aggregated:
            aggregated[name] /= K

        # DP: add noise via DP-FTRL tree mechanism
        if self.dp_enabled:
            if self.dp_mode == "ftrl" and self.tree_agg is not None:
                self.tree_agg.add_to_tree(self.server_version, aggregated)
                noise = self.tree_agg.get_sum(self.server_version)
                for name in aggregated:
                    aggregated[name] += noise[name]
            elif self.dp_mode == "sgd":
                aggregated = add_gaussian_noise(
                    aggregated, self.noise_scale, self.clip_norm, K
                )

        # Apply server update: w_{t+1} = w_t - η_g * Δ_t
        self.apply_update(aggregated)

        # Reset buffer
        self.buffer.clear()

    def get_buffer_occupancy(self) -> int:
        return len(self.buffer)
