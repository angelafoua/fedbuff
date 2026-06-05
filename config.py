"""
Experiment configurations reproducing Table 1 and Table 4 from the paper.

Hyperparameters were tuned via Bayesian optimization (Snoek et al., 2012).
Best values are from Table 4 in Appendix B.3.2.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class ExperimentConfig:
    # Algorithm
    algorithm: str = "fedbuff"  # fedbuff, fedasync, fedavgm, fedavg, fedprox

    # Dataset
    dataset: str = "cifar10"  # cifar10, sent140, celeba
    data_dir: str = "./data"

    # FL parameters
    concurrency: int = 1000       # Max clients training in parallel
    buffer_size: int = 10         # K for FedBuff
    max_client_trips: int = 600000  # Budget (600K as in paper)
    eval_every_trips: int = 5000  # Evaluate every N client trips

    # Optimizer
    client_lr: float = 1.95e-4    # η_l
    server_lr: float = 40.9       # η_g
    momentum: float = 0.0         # β (server momentum)
    proximal_mu: float = 0.0      # μ (FedProx)
    batch_size: int = 32          # B

    # FedBuff specific
    staleness_alpha: float = 0.5  # Staleness scaling exponent
    lr_norm: bool = True          # Learning rate normalization

    # Delay simulation
    delay_distribution: str = "half_normal"  # half_normal, uniform, exponential
    delay_scale: float = 10.0

    # Differential Privacy
    dp_enabled: bool = False
    dp_mode: str = "ftrl"         # ftrl or sgd
    clip_norm: float = 1.0
    noise_scale: float = 0.0
    dp_delta: float = 1e-7

    # General
    device: str = "cpu"
    num_seeds: int = 3
    seed: int = 0
    target_accuracy: float = 60.0  # Target validation accuracy (%)
    log_dir: str = "./logs"


# ============================================================
# Best hyperparameters from Table 4 (Appendix B.3.2)
# ============================================================

BEST_HYPERPARAMS: Dict[str, Dict[str, Dict]] = {
    "cifar10": {
        "fedbuff":  {"client_lr": 1.95e-4, "server_lr": 40.9,  "momentum": 0.0, "target_accuracy": 60.0},
        "fedasync": {"client_lr": 1e2,     "server_lr": 6.4e-5, "momentum": 0.0, "staleness_alpha": 0.5, "target_accuracy": 60.0},
        "fedavgm":  {"client_lr": 1e1,     "server_lr": 1.02e-3, "momentum": 0.9, "target_accuracy": 60.0},
        "fedavg":   {"client_lr": 1e1,     "server_lr": 1.02e-3, "momentum": 0.0, "target_accuracy": 60.0},
        "fedprox":  {"client_lr": 1e1,     "server_lr": 1.02e-3, "momentum": 0.0, "proximal_mu": 1e-3, "target_accuracy": 60.0},
    },
    "celeba": {
        "fedbuff":  {"client_lr": 4.7e-6,  "server_lr": 1e3,    "momentum": 0.3, "target_accuracy": 90.0},
        "fedasync": {"client_lr": 5.7,     "server_lr": 2.8e-3, "momentum": 0.0, "staleness_alpha": 0.5, "target_accuracy": 90.0},
        "fedavgm":  {"client_lr": 1.1e-1,  "server_lr": 2.4e-1, "momentum": 0.83, "target_accuracy": 90.0},
        "fedavg":   {"client_lr": 1e2,     "server_lr": 1.6e-3, "momentum": 0.0, "target_accuracy": 90.0},
        "fedprox":  {"client_lr": 4.9e-4,  "server_lr": 1e2,    "momentum": 0.0, "proximal_mu": 1e-2, "target_accuracy": 90.0},
    },
    "sent140": {
        "fedbuff":  {"client_lr": 13.0,    "server_lr": 4.9e-2, "momentum": 0.5, "target_accuracy": 69.0},
        "fedasync": {"client_lr": 17.0,    "server_lr": 1.5e-2, "momentum": 0.0, "staleness_alpha": 0.5, "target_accuracy": 69.0},
        "fedavgm":  {"client_lr": 1.5,     "server_lr": 3.4e-1, "momentum": 0.9, "target_accuracy": 69.0},
        "fedavg":   {"client_lr": 2.6e-3,  "server_lr": 1e3,    "momentum": 0.0, "target_accuracy": 69.0},
        "fedprox":  {"client_lr": 2.0e-3,  "server_lr": 1e3,    "momentum": 0.0, "proximal_mu": 1e-3, "target_accuracy": 69.0},
    },
}

# DP hyperparameters from Table 3
DP_HYPERPARAMS: Dict[str, Dict[str, Dict]] = {
    "sent140": {
        "fedbuff_ftrl_eps6":  {"client_lr": 1.0,   "server_lr": 4.3,     "momentum": 0.99,  "clip_norm": 1.2e-4},
        "fedbuff_ftrl_eps12": {"client_lr": 1e-2,   "server_lr": 54.0,    "momentum": 0.0,   "clip_norm": 1.1e-3},
        "fedbuff_ftrl_eps24": {"client_lr": 1e-1,   "server_lr": 870.0,   "momentum": 0.3,   "clip_norm": 1e-4},
        "syncfl_dpsgd_eps6":  {"client_lr": 1e-1,   "server_lr": 590.0,   "momentum": 0.3,   "clip_norm": 1.1e-2},
        "syncfl_dpsgd_eps12": {"client_lr": 1.0,    "server_lr": 1e4,     "momentum": 0.5,   "clip_norm": 7.6e-3},
        "syncfl_dpsgd_eps24": {"client_lr": 1e-1,   "server_lr": 510.0,   "momentum": 0.3,   "clip_norm": 1.4e-2},
        "syncfl_ftrl_eps6":   {"client_lr": 1e-3,   "server_lr": 2.6e4,   "momentum": 0.1,   "clip_norm": 2.7e-4},
        "syncfl_ftrl_eps12":  {"client_lr": 1.0,    "server_lr": 100.0,   "momentum": 0.9,   "clip_norm": 2.7e-1},
        "syncfl_ftrl_eps24":  {"client_lr": 1.0,    "server_lr": 8e3,     "momentum": 0.5,   "clip_norm": 1e-4},
    },
}


def get_config(algorithm: str, dataset: str, concurrency: int = 1000,
               buffer_size: int = 10, seed: int = 0, **overrides) -> ExperimentConfig:
    """Create an ExperimentConfig with best hyperparameters from the paper."""
    cfg = ExperimentConfig(
        algorithm=algorithm,
        dataset=dataset,
        concurrency=concurrency,
        buffer_size=buffer_size,
        seed=seed,
    )

    # Apply best hyperparameters
    if dataset in BEST_HYPERPARAMS and algorithm in BEST_HYPERPARAMS[dataset]:
        best = BEST_HYPERPARAMS[dataset][algorithm]
        for k, v in best.items():
            setattr(cfg, k, v)

    # Apply overrides
    for k, v in overrides.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)

    return cfg
