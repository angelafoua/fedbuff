"""
FedAsync: Fully Asynchronous FL (Xie et al., 2019).

Every single client update triggers a server model update (K=1 effectively).
This is incompatible with SecAgg (no aggregation to hide individual updates).
Uses staleness-weighted updates similar to FedBuff.
"""

from typing import List
import torch
from .base import FLServer, ClientUpdate
from ..utils.delay import staleness_scaling


class FedAsyncServer(FLServer):
    """FedAsync server — updates on every client return.
    
    Key difference from FedBuff: K=1 (no buffering).
    This means every client update immediately triggers a server step.
    """

    def __init__(self, model, server_lr: float = 1.0, momentum: float = 0.0,
                 staleness_alpha: float = 0.5, device: str = "cpu"):
        super().__init__(model, server_lr, momentum, device)
        self.staleness_alpha = staleness_alpha

    def process_updates(self, updates: List[ClientUpdate]) -> bool:
        """Process updates one at a time (fully async)."""
        updated = False
        for update in updates:
            self.total_client_trips += 1
            weight = staleness_scaling(update.staleness, self.staleness_alpha)

            # Scale the update by staleness weight
            weighted_delta = {
                name: weight * delta
                for name, delta in update.delta.items()
            }

            # Immediate server update for every single client
            self.apply_update(weighted_delta)
            updated = True

        return updated
