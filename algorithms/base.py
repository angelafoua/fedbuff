"""
Base classes for FL server and client.

Provides the common interface used by all FL algorithms:
FedBuff, FedAvg, FedAvgM, FedProx, FedAsync.
"""

import copy
from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class ClientUpdate:
    """Represents a single client's update to the server."""

    def __init__(self, client_id: int, delta: Dict[str, torch.Tensor],
                 num_samples: int, download_version: int,
                 staleness: int = 0, delay: float = 0.0):
        self.client_id = client_id
        self.delta = delta  # w_download - w_after_local_training (pseudo-gradient)
        self.num_samples = num_samples
        self.download_version = download_version
        self.staleness = staleness
        self.delay = delay


class FLClient:
    """Simulates a single FL client performing local training.
    
    Each client:
    1. Downloads the current server model
    2. Performs one epoch of local SGD on its data
    3. Returns the update (delta) to the server
    """

    def __init__(self, client_id: int, dataset: Dataset,
                 batch_size: int = 32, device: str = "cpu"):
        self.client_id = client_id
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device
        self.num_samples = len(dataset)

    def local_train(self, model: nn.Module, lr: float,
                    server_batch_size: int = 32,
                    proximal_mu: float = 0.0,
                    lr_norm: bool = False) -> Dict[str, torch.Tensor]:
        """Perform one epoch of local SGD and return the update delta.
        
        Args:
            model: Copy of the server model to train locally.
            lr: Client learning rate η_l.
            server_batch_size: Prescribed batch size B (for LR normalization).
            proximal_mu: FedProx proximal term μ (0 = no proximal).
            lr_norm: Whether to apply learning rate normalization (Section 5).
        
        Returns:
            Delta dict: {param_name: w_initial - w_trained} (pseudo-gradient)
        """
        model = model.to(self.device)
        model.train()

        # Save initial parameters
        initial_params = {name: p.data.clone() for name, p in model.named_parameters()}

        # DataLoader for one epoch
        loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)
        criterion = nn.CrossEntropyLoss()

        for batch in loader:
            if len(batch) == 2:
                inputs, targets = batch
            else:
                inputs, targets = batch[0], batch[1]

            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            actual_batch_size = inputs.size(0)

            # Learning rate normalization (Section 5)
            if lr_norm and actual_batch_size < server_batch_size:
                effective_lr = lr * actual_batch_size / server_batch_size
            else:
                effective_lr = lr

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # FedProx proximal term
            if proximal_mu > 0:
                prox_loss = 0.0
                for name, p in model.named_parameters():
                    prox_loss += torch.sum((p - initial_params[name]) ** 2)
                loss += (proximal_mu / 2) * prox_loss

            # Backward pass + manual SGD step
            model.zero_grad()
            loss.backward()

            with torch.no_grad():
                for name, p in model.named_parameters():
                    if p.grad is not None:
                        p.data -= effective_lr * p.grad

        # Compute delta = initial - trained (pseudo-gradient direction)
        delta = {}
        with torch.no_grad():
            for name, p in model.named_parameters():
                delta[name] = initial_params[name] - p.data

        return delta


class FLServer(ABC):
    """Abstract base class for FL server algorithms."""

    def __init__(self, model: nn.Module, server_lr: float = 1.0,
                 momentum: float = 0.0, device: str = "cpu"):
        self.model = model.to(device)
        self.server_lr = server_lr  # η_g
        self.momentum = momentum    # β
        self.device = device
        self.server_version = 0
        self.total_client_trips = 0

        # Momentum buffer
        self.momentum_buffer: Optional[Dict[str, torch.Tensor]] = None

    def get_model_copy(self) -> nn.Module:
        """Return a deep copy of the current server model."""
        return copy.deepcopy(self.model)

    def get_model_state(self) -> Dict[str, torch.Tensor]:
        """Return current model parameters."""
        return {name: p.data.clone() for name, p in self.model.named_parameters()}

    def apply_update(self, aggregated_delta: Dict[str, torch.Tensor]):
        """Apply aggregated update to the server model.
        
        w_{t+1} = w_t - η_g * Δ_t
        
        With optional server-side momentum (FedAvgM):
        v_{t+1} = β * v_t + Δ_t
        w_{t+1} = w_t - η_g * v_{t+1}
        """
        with torch.no_grad():
            if self.momentum > 0:
                if self.momentum_buffer is None:
                    self.momentum_buffer = {
                        name: delta.clone()
                        for name, delta in aggregated_delta.items()
                    }
                else:
                    for name in self.momentum_buffer:
                        self.momentum_buffer[name] = (
                            self.momentum * self.momentum_buffer[name] +
                            aggregated_delta[name]
                        )

                for name, p in self.model.named_parameters():
                    p.data -= self.server_lr * self.momentum_buffer[name]
            else:
                for name, p in self.model.named_parameters():
                    p.data -= self.server_lr * aggregated_delta[name]

        self.server_version += 1

    @abstractmethod
    def process_updates(self, updates: List[ClientUpdate]) -> bool:
        """Process received client updates. Returns True if server model was updated."""
        pass
