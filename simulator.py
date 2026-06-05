"""
Core FL Simulator.

Simulates both synchronous and asynchronous federated learning.
Handles client scheduling, delay simulation, and training orchestration.

Key design: uses "client trips" as the universal metric (Section 6),
measuring both communication and computation cost.
"""

import time
import copy
import numpy as np
from typing import List, Optional, Tuple
from collections import deque

import torch
import torch.nn as nn

from .config import ExperimentConfig
from .algorithms.base import FLClient, FLServer, ClientUpdate
from .algorithms.fedbuff import FedBuffServer
from .algorithms.fedasync import FedAsyncServer
from .algorithms.fedavg import SyncFLServer, create_sync_server
from .utils.delay import DelaySimulator, StalenessTracker
from .utils.metrics import ExperimentMetrics, evaluate_model, print_experiment_summary
from .utils.models import get_model


class FLSimulator:
    """Federated Learning Simulator.
    
    Supports:
    - Synchronous FL (FedAvg, FedAvgM, FedProx)
    - Asynchronous FL (FedAsync, FedBuff)
    - Configurable delay distributions
    - Client trip counting for fair comparison
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device = config.device

        # Set seeds
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        # Initialize dataset
        self.fl_dataset = self._create_dataset()
        self.num_clients = self.fl_dataset.num_actual_clients

        # Initialize model
        self.model = self._create_model()

        # Initialize server
        self.server = self._create_server()

        # Initialize delay simulator
        self.delay_sim = DelaySimulator(
            distribution=config.delay_distribution,
            scale=config.delay_scale,
            seed=config.seed
        )

        # Staleness tracking for async methods
        self.staleness_tracker = StalenessTracker()

        # Metrics
        self.metrics = ExperimentMetrics(
            algorithm=config.algorithm,
            dataset=config.dataset,
            concurrency=config.concurrency,
            buffer_size=config.buffer_size if config.algorithm == "fedbuff" else 1,
            seed=config.seed,
            target_accuracy=config.target_accuracy,
        )

        # Validation/test loaders
        self.val_loader = self.fl_dataset.get_val_loader(batch_size=128)
        self.test_loader = self.fl_dataset.get_test_loader(batch_size=128)

    def _create_dataset(self):
        cfg = self.config
        if cfg.dataset == "cifar10":
            from .datasets.cifar10 import CIFAR10FL
            return CIFAR10FL(data_dir=cfg.data_dir, seed=cfg.seed)
        elif cfg.dataset == "sent140":
            from .datasets.leaf import Sent140FL
            return Sent140FL(data_dir=f"{cfg.data_dir}/sent140")
        elif cfg.dataset == "celeba":
            from .datasets.leaf import CelebAFL
            return CelebAFL(data_dir=f"{cfg.data_dir}/celeba")
        else:
            raise ValueError(f"Unknown dataset: {cfg.dataset}")

    def _create_model(self) -> nn.Module:
        cfg = self.config
        if cfg.dataset == "cifar10":
            return get_model("cifar10").to(self.device)
        elif cfg.dataset == "celeba":
            return get_model("celeba").to(self.device)
        elif cfg.dataset == "sent140":
            return get_model("sent140").to(self.device)
        else:
            raise ValueError(f"Unknown dataset: {cfg.dataset}")

    def _create_server(self) -> FLServer:
        cfg = self.config
        model_copy = copy.deepcopy(self.model)

        if cfg.algorithm == "fedbuff":
            return FedBuffServer(
                model_copy, server_lr=cfg.server_lr, momentum=cfg.momentum,
                buffer_size=cfg.buffer_size, staleness_alpha=cfg.staleness_alpha,
                lr_norm=cfg.lr_norm, device=self.device,
                dp_enabled=cfg.dp_enabled, clip_norm=cfg.clip_norm,
                noise_scale=cfg.noise_scale, dp_mode=cfg.dp_mode,
            )
        elif cfg.algorithm == "fedasync":
            return FedAsyncServer(
                model_copy, server_lr=cfg.server_lr, momentum=cfg.momentum,
                staleness_alpha=cfg.staleness_alpha, device=self.device,
            )
        elif cfg.algorithm in ("fedavg", "fedavgm", "fedprox"):
            return create_sync_server(
                cfg.algorithm, model_copy, server_lr=cfg.server_lr,
                momentum=cfg.momentum, device=self.device,
            )
        else:
            raise ValueError(f"Unknown algorithm: {cfg.algorithm}")

    def _sample_client(self) -> int:
        """Sample a random client to participate."""
        return np.random.randint(0, self.num_clients)

    def _train_client(self, client_id: int, model: nn.Module) -> ClientUpdate:
        """Simulate one client performing local training."""
        client_dataset = self.fl_dataset.get_client_dataset(client_id)
        client = FLClient(client_id, client_dataset,
                          batch_size=self.config.batch_size, device=self.device)

        delta = client.local_train(
            model, lr=self.config.client_lr,
            server_batch_size=self.config.batch_size,
            proximal_mu=self.config.proximal_mu if self.config.algorithm == "fedprox" else 0.0,
            lr_norm=self.config.lr_norm,
        )

        delay = self.delay_sim.sample_one()
        download_version = self.staleness_tracker.record_download()

        return ClientUpdate(
            client_id=client_id,
            delta=delta,
            num_samples=client.num_samples,
            download_version=download_version,
            delay=delay,
        )

    def run(self) -> ExperimentMetrics:
        """Run the full FL simulation."""
        cfg = self.config
        start_time = time.time()
        total_trips = 0
        server_updates = 0
        last_eval_trips = 0

        print(f"\n{'='*60}")
        print(f"Running {cfg.algorithm.upper()} on {cfg.dataset.upper()}")
        print(f"  Concurrency={cfg.concurrency}, K={cfg.buffer_size}, "
              f"η_l={cfg.client_lr}, η_g={cfg.server_lr}, β={cfg.momentum}")
        print(f"  Target: {cfg.target_accuracy}% val accuracy")
        print(f"  Budget: {cfg.max_client_trips:,} client trips")
        print(f"{'='*60}")

        # Initial evaluation
        val_acc, val_loss = evaluate_model(self.server.model, self.val_loader, self.device)
        self.metrics.record(0, 0, 0.0, val_acc, val_loss, 0.0)
        print(f"  [Trip 0] Val Acc: {val_acc:.2f}%")

        if cfg.algorithm in ("fedavg", "fedavgm", "fedprox"):
            self._run_sync(start_time)
        else:
            self._run_async(start_time)

        self.metrics.finalize()
        print_experiment_summary(self.metrics)
        return self.metrics

    def _run_sync(self, start_time: float):
        """Run synchronous FL training loop."""
        cfg = self.config
        total_trips = 0

        while total_trips < cfg.max_client_trips:
            # One synchronous round: sample cohort, all train, aggregate
            cohort_size = cfg.concurrency
            model_copy = self.server.get_model_copy()
            updates = []

            for _ in range(cohort_size):
                client_id = self._sample_client()
                update = self._train_client(client_id, copy.deepcopy(model_copy))
                update.staleness = 0  # No staleness in sync FL
                updates.append(update)

            self.server.process_updates(updates)
            total_trips += cohort_size
            self.staleness_tracker.server_step()

            # Evaluate periodically
            if total_trips - self._last_eval_trips() >= cfg.eval_every_trips:
                self._evaluate_and_log(total_trips, start_time)

            # Check budget
            if self.metrics.trips_to_target is not None:
                # Keep running a bit past target for full curves
                if total_trips > self.metrics.trips_to_target * 1.5:
                    break

    def _run_async(self, start_time: float):
        """Run asynchronous FL training loop.
        
        Simulates async client arrivals at constant rate.
        Clients download model, train locally (with delay), upload update.
        """
        cfg = self.config
        total_trips = 0

        # Active clients: (finish_time, client_id, download_version, model_snapshot)
        active_clients: List = []
        current_time = 0.0

        while total_trips < cfg.max_client_trips:
            # Fill up to concurrency
            while len(active_clients) < cfg.concurrency and total_trips < cfg.max_client_trips:
                client_id = self._sample_client()
                download_version = self.staleness_tracker.record_download()
                delay = self.delay_sim.sample_one()
                finish_time = current_time + delay

                active_clients.append({
                    'finish_time': finish_time,
                    'client_id': client_id,
                    'download_version': download_version,
                    'model_snapshot': self.server.get_model_copy(),
                })

            if not active_clients:
                break

            # Process the earliest finishing client
            active_clients.sort(key=lambda x: x['finish_time'])
            earliest = active_clients.pop(0)
            current_time = earliest['finish_time']

            # Train client with the model they downloaded
            client_dataset = self.fl_dataset.get_client_dataset(earliest['client_id'])
            client = FLClient(earliest['client_id'], client_dataset,
                              batch_size=cfg.batch_size, device=self.device)

            delta = client.local_train(
                earliest['model_snapshot'], lr=cfg.client_lr,
                server_batch_size=cfg.batch_size, lr_norm=cfg.lr_norm,
            )

            staleness = self.staleness_tracker.compute_staleness(earliest['download_version'])

            update = ClientUpdate(
                client_id=earliest['client_id'],
                delta=delta,
                num_samples=client.num_samples,
                download_version=earliest['download_version'],
                staleness=staleness,
                delay=earliest['finish_time'] - current_time,
            )

            # Process update
            if isinstance(self.server, FedBuffServer):
                updated = self.server.receive_update(update)
            else:
                updated = self.server.process_updates([update])

            if updated:
                self.staleness_tracker.server_step()

            total_trips += 1

            # Evaluate periodically
            if total_trips - self._last_eval_trips() >= cfg.eval_every_trips:
                self._evaluate_and_log(total_trips, start_time)

    def _last_eval_trips(self) -> int:
        if self.metrics.client_trips:
            return self.metrics.client_trips[-1]
        return 0

    def _evaluate_and_log(self, total_trips: int, start_time: float):
        """Evaluate current model and log metrics."""
        val_acc, val_loss = evaluate_model(self.server.model, self.val_loader, self.device)
        wall_clock = time.time() - start_time

        self.metrics.record(
            client_trips=total_trips,
            server_update=self.server.server_version,
            train_acc=0.0,  # Skip train eval for speed
            val_acc=val_acc,
            train_loss=val_loss,
            wall_clock=wall_clock,
        )

        status = "✓ TARGET" if self.metrics.trips_to_target == total_trips else ""
        print(f"  [Trip {total_trips:>7,}] Val Acc: {val_acc:.2f}% | "
              f"Server v{self.server.server_version} | "
              f"Time: {wall_clock:.0f}s {status}")
