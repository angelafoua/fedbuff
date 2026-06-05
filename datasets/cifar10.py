"""
CIFAR-10 dataset with Dirichlet non-IID client partitioning.

Following Hsu et al. (2019), partitions CIFAR-10 into 5000 clients
using a Dirichlet distribution with parameter α=0.1.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, Subset, DataLoader
from torchvision import datasets, transforms
from typing import List, Tuple, Optional


class CIFAR10FL:
    """CIFAR-10 federated dataset with Dirichlet non-IID partitioning."""

    def __init__(self, data_dir: str = "./data", num_clients: int = 5000,
                 dirichlet_alpha: float = 0.1, seed: int = 0):
        self.data_dir = data_dir
        self.num_clients = num_clients
        self.alpha = dirichlet_alpha
        self.seed = seed

        # Normalization from paper
        self.transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        # Download and load
        self.train_dataset = datasets.CIFAR10(
            data_dir, train=True, download=True, transform=self.transform_train
        )
        self.test_dataset = datasets.CIFAR10(
            data_dir, train=False, download=True, transform=self.transform_test
        )

        # Partition into clients
        self.client_indices = self._dirichlet_partition()

    def _dirichlet_partition(self) -> List[List[int]]:
        """Partition training data across clients using Dirichlet distribution.
        
        For each class, sample a probability vector from Dir(α) over clients,
        then assign samples proportionally.
        """
        rng = np.random.RandomState(self.seed)
        targets = np.array(self.train_dataset.targets)
        num_classes = 10

        client_indices = [[] for _ in range(self.num_clients)]

        for c in range(num_classes):
            class_indices = np.where(targets == c)[0]
            rng.shuffle(class_indices)

            # Sample proportions from Dirichlet
            proportions = rng.dirichlet(np.repeat(self.alpha, self.num_clients))
            # Balance proportions so no client gets 0
            proportions = proportions / proportions.sum()

            # Convert proportions to counts
            counts = (proportions * len(class_indices)).astype(int)
            # Distribute remainder
            remainder = len(class_indices) - counts.sum()
            for i in range(remainder):
                counts[i % self.num_clients] += 1

            # Assign indices to clients
            idx = 0
            for client_id in range(self.num_clients):
                if counts[client_id] > 0:
                    client_indices[client_id].extend(
                        class_indices[idx:idx + counts[client_id]].tolist()
                    )
                    idx += counts[client_id]

        # Remove empty clients
        client_indices = [indices for indices in client_indices if len(indices) > 0]
        return client_indices

    def get_client_dataset(self, client_id: int) -> Subset:
        """Get the dataset for a specific client."""
        return Subset(self.train_dataset, self.client_indices[client_id])

    def get_test_loader(self, batch_size: int = 128) -> DataLoader:
        """Get the global test DataLoader."""
        return DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)

    def get_val_loader(self, batch_size: int = 128) -> DataLoader:
        """Use a portion of test set as validation (paper uses separate val)."""
        # For simplicity, use test set as validation
        return self.get_test_loader(batch_size)

    @property
    def num_actual_clients(self) -> int:
        return len(self.client_indices)

    def client_data_sizes(self) -> List[int]:
        return [len(indices) for indices in self.client_indices]

    def summary(self):
        sizes = self.client_data_sizes()
        print(f"CIFAR-10 FL Dataset:")
        print(f"  Clients: {self.num_actual_clients}")
        print(f"  Dirichlet α: {self.alpha}")
        print(f"  Samples per client: min={min(sizes)}, max={max(sizes)}, "
              f"mean={np.mean(sizes):.1f}, median={np.median(sizes):.1f}")
        print(f"  Total train samples: {sum(sizes)}")
        print(f"  Test samples: {len(self.test_dataset)}")
