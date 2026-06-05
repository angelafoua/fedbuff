"""
Evaluation metrics and logging for FL experiments.

Key metric from the paper: number of client trips to reach target accuracy.
"""

import time
import json
import os
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@dataclass
class ExperimentMetrics:
    """Tracks all metrics for an FL experiment run."""
    algorithm: str = ""
    dataset: str = ""
    concurrency: int = 0
    buffer_size: int = 1  # K
    seed: int = 0
    target_accuracy: float = 0.0

    # Per-evaluation metrics
    client_trips: List[int] = field(default_factory=list)
    train_accuracies: List[float] = field(default_factory=list)
    val_accuracies: List[float] = field(default_factory=list)
    test_accuracies: List[float] = field(default_factory=list)
    train_losses: List[float] = field(default_factory=list)
    server_updates: List[int] = field(default_factory=list)
    wall_clock_times: List[float] = field(default_factory=list)

    # Summary
    trips_to_target: Optional[int] = None
    final_accuracy: float = 0.0
    total_client_trips: int = 0
    total_wall_clock: float = 0.0

    def record(self, client_trips: int, server_update: int,
               train_acc: float, val_acc: float, train_loss: float,
               wall_clock: float, test_acc: float = 0.0):
        self.client_trips.append(client_trips)
        self.server_updates.append(server_update)
        self.train_accuracies.append(train_acc)
        self.val_accuracies.append(val_acc)
        self.test_accuracies.append(test_acc)
        self.train_losses.append(train_loss)
        self.wall_clock_times.append(wall_clock)

        # Check if target accuracy reached for the first time
        if self.trips_to_target is None and val_acc >= self.target_accuracy:
            self.trips_to_target = client_trips

    def finalize(self):
        if self.val_accuracies:
            self.final_accuracy = self.val_accuracies[-1]
        if self.client_trips:
            self.total_client_trips = self.client_trips[-1]
        if self.wall_clock_times:
            self.total_wall_clock = self.wall_clock_times[-1]

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'ExperimentMetrics':
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)


def evaluate_model(model: nn.Module, dataloader: DataLoader,
                   device: str = "cpu") -> tuple:
    """Evaluate model accuracy and loss on a dataset.
    
    Returns:
        (accuracy, average_loss)
    """
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 2:
                inputs, targets = batch
            else:
                inputs, targets = batch[0], batch[1]

            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = 100.0 * correct / total if total > 0 else 0.0
    avg_loss = total_loss / total if total > 0 else 0.0
    return accuracy, avg_loss


def print_experiment_summary(metrics: ExperimentMetrics):
    """Print a formatted summary of experiment results."""
    print("\n" + "=" * 60)
    print(f"Experiment Summary: {metrics.algorithm} on {metrics.dataset}")
    print("=" * 60)
    print(f"  Concurrency:     {metrics.concurrency}")
    print(f"  Buffer size K:   {metrics.buffer_size}")
    print(f"  Seed:            {metrics.seed}")
    print(f"  Target accuracy: {metrics.target_accuracy}%")
    print(f"  Final val acc:   {metrics.final_accuracy:.2f}%")
    print(f"  Total trips:     {metrics.total_client_trips:,}")

    if metrics.trips_to_target is not None:
        print(f"  Trips to target: {metrics.trips_to_target:,}")
    else:
        print(f"  Trips to target: NOT REACHED")

    print(f"  Wall-clock time: {metrics.total_wall_clock:.1f}s")
    print("=" * 60)


def format_table1(results: Dict[str, List[ExperimentMetrics]]):
    """Format results as Table 1 from the paper."""
    print("\n" + "=" * 80)
    print("Table 1: Client trips (×1000) to reach target accuracy")
    print("=" * 80)
    header = f"{'Dataset':<12} {'Accuracy':<10} {'FedBuff':<12} {'FedAsync':<15} {'FedAvgM':<15} {'FedAvg':<15} {'FedProx':<15}"
    print(header)
    print("-" * 80)

    for dataset_key, metrics_list in results.items():
        row = {}
        for m in metrics_list:
            trips = m.trips_to_target
            if trips is not None:
                row[m.algorithm] = f"{trips / 1000:.1f}"
            else:
                row[m.algorithm] = "> budget"

        target = metrics_list[0].target_accuracy if metrics_list else "?"
        algos = ["fedbuff", "fedasync", "fedavgm", "fedavg", "fedprox"]
        vals = [row.get(a, "N/A") for a in algos]
        print(f"{dataset_key:<12} {target}%{'':<6} {'  '.join(f'{v:<12}' for v in vals)}")
