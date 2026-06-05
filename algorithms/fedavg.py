"""
Synchronous FL baselines: FedAvg, FedAvgM, FedProx.

In SyncFL, the server waits for ALL clients in the cohort to finish
before aggregating and updating. Concurrency = cohort size = clients-per-round.

- FedAvg (McMahan et al., 2016): vanilla local SGD + averaging
- FedAvgM (Hsu et al., 2019): FedAvg + server-side momentum
- FedProx (Li et al., 2018): FedAvg + proximal term on client
"""

from typing import List, Dict
import torch
from algorithms.base import FLServer, ClientUpdate


class SyncFLServer(FLServer):
    """Synchronous FL server (FedAvg / FedAvgM / FedProx).
    
    All three share the same server aggregation; they differ only in:
    - FedAvg: momentum=0, proximal_mu=0
    - FedAvgM: momentum>0, proximal_mu=0
    - FedProx: momentum=0, proximal_mu>0 (proximal term applied at client)
    
    The server aggregates all cohort updates uniformly:
    Δ_t = (1/|S_t|) * Σ_{i ∈ S_t} Δ_i
    """

    def __init__(self, model, server_lr: float = 1.0, momentum: float = 0.0,
                 device: str = "cpu", over_selection_ratio: float = 0.0):
        super().__init__(model, server_lr, momentum, device)
        self.over_selection_ratio = over_selection_ratio  # e.g., 0.3 for 30%

    def process_updates(self, updates: List[ClientUpdate]) -> bool:
        """Process a full cohort of synchronous updates.
        
        In SyncFL, all updates in the cohort arrive together (no staleness).
        With over-selection, only the fastest (1 - over_selection_ratio) are used.
        """
        if not updates:
            return False

        self.total_client_trips += len(updates)

        # Over-selection: sort by delay, keep fastest
        if self.over_selection_ratio > 0:
            updates = sorted(updates, key=lambda u: u.delay)
            keep = max(1, int(len(updates) * (1 - self.over_selection_ratio)))
            updates = updates[:keep]

        # Simple uniform averaging: Δ_t = (1/K) * Σ Δ_i
        K = len(updates)
        aggregated: Dict[str, torch.Tensor] = {}

        for update in updates:
            for name, delta in update.delta.items():
                if name not in aggregated:
                    aggregated[name] = torch.zeros_like(delta)
                aggregated[name] += delta

        for name in aggregated:
            aggregated[name] /= K

        self.apply_update(aggregated)
        return True


def create_sync_server(algorithm: str, model, server_lr: float,
                       momentum: float = 0.0, proximal_mu: float = 0.0,
                       device: str = "cpu", **kwargs) -> SyncFLServer:
    """Factory for creating synchronous FL servers.
    
    Args:
        algorithm: One of 'fedavg', 'fedavgm', 'fedprox'.
        model: Neural network model.
        server_lr: Server learning rate η_g.
        momentum: Server momentum β (only for FedAvgM).
        proximal_mu: Proximal term μ (only for FedProx, applied at client).
    """
    if algorithm == "fedavg":
        return SyncFLServer(model, server_lr=server_lr, momentum=0.0, device=device)
    elif algorithm == "fedavgm":
        return SyncFLServer(model, server_lr=server_lr, momentum=momentum, device=device)
    elif algorithm == "fedprox":
        # Note: proximal_mu is applied on the client side during local training
        return SyncFLServer(model, server_lr=server_lr, momentum=0.0, device=device)
    else:
        raise ValueError(f"Unknown sync algorithm: {algorithm}")
