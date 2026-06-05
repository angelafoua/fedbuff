"""
Main experiment runner for FedBuff reproduction.

Usage:
    # Single experiment
    python run_experiment.py --algorithm fedbuff --dataset cifar10 --concurrency 100 --K 10

    # Reproduce Table 1 (all algorithms, all datasets)
    python run_experiment.py --run_table1

    # Reproduce Figure 3 (scalability across concurrency levels)
    python run_experiment.py --run_scalability --dataset cifar10

    # Reproduce Table 2 (effect of K)
    python run_experiment.py --run_buffer_sweep --dataset cifar10

    # DP experiment (Figure 4)
    python run_experiment.py --run_dp --dataset sent140
"""

import argparse
import os
import json
import numpy as np
from typing import List, Dict

from config import ExperimentConfig, get_config, BEST_HYPERPARAMS
from simulator import FLSimulator
from utils.metrics import ExperimentMetrics, format_table1


def run_single(config: ExperimentConfig) -> ExperimentMetrics:
    """Run a single FL experiment."""
    simulator = FLSimulator(config)
    metrics = simulator.run()

    # Save results
    save_path = os.path.join(
        config.log_dir,
        f"{config.dataset}_{config.algorithm}_C{config.concurrency}_K{config.buffer_size}_s{config.seed}.json"
    )
    metrics.save(save_path)
    print(f"Results saved to {save_path}")

    return metrics


def run_multi_seed(config: ExperimentConfig) -> List[ExperimentMetrics]:
    """Run experiment with multiple seeds and report average ± std."""
    results = []
    for seed in range(config.num_seeds):
        cfg = ExperimentConfig(**vars(config))
        cfg.seed = seed
        metrics = run_single(cfg)
        results.append(metrics)

    # Report average
    trips = [m.trips_to_target for m in results if m.trips_to_target is not None]
    if trips:
        mean_trips = np.mean(trips)
        std_trips = np.std(trips)
        print(f"\n{'='*40}")
        print(f"Average over {len(trips)} seeds:")
        print(f"  Trips to target: {mean_trips/1000:.1f}K ± {std_trips/1000:.1f}K")
        print(f"{'='*40}")
    else:
        print(f"\nTarget accuracy NOT reached in any seed.")

    return results


def run_table1(args):
    """Reproduce Table 1: comparison of all methods at concurrency=1000."""
    print("\n" + "=" * 80)
    print("REPRODUCING TABLE 1: Average client trips to target accuracy")
    print("=" * 80)

    datasets = ["celeba", "sent140", "cifar10"]
    algorithms = ["fedbuff", "fedasync", "fedavgm", "fedavg", "fedprox"]
    all_results: Dict[str, List[ExperimentMetrics]] = {}

    for dataset in datasets:
        dataset_results = []
        for algo in algorithms:
            print(f"\n>>> Running {algo} on {dataset} <<<")
            config = get_config(
                algorithm=algo, dataset=dataset,
                concurrency=1000,
                buffer_size=10 if algo == "fedbuff" else 1,
                seed=0,
                device=args.device,
                data_dir=args.data_dir,
                log_dir=args.log_dir,
            )
            metrics = run_single(config)
            dataset_results.append(metrics)

        all_results[dataset] = dataset_results

    format_table1(all_results)


def run_scalability(args):
    """Reproduce Figure 3: scalability across concurrency levels."""
    print("\n" + "=" * 80)
    print("REPRODUCING FIGURE 3: Scalability of FedBuff vs FedAvgM")
    print("=" * 80)

    concurrency_levels = [10, 50, 100, 200, 500, 1000]
    dataset = args.dataset

    for algo in ["fedbuff", "fedavgm"]:
        print(f"\n--- {algo.upper()} ---")
        for conc in concurrency_levels:
            config = get_config(
                algorithm=algo, dataset=dataset,
                concurrency=conc,
                buffer_size=10 if algo == "fedbuff" else conc,
                seed=0,
                device=args.device,
                data_dir=args.data_dir,
                log_dir=args.log_dir,
            )
            metrics = run_single(config)
            trips = metrics.trips_to_target
            trips_str = f"{trips/1000:.1f}K" if trips else "> budget"
            print(f"  Concurrency={conc:>5}: {trips_str}")


def run_buffer_sweep(args):
    """Reproduce Table 2: effect of buffer size K."""
    print("\n" + "=" * 80)
    print("REPRODUCING TABLE 2: FedBuff with different K values")
    print("=" * 80)

    K_values = [1, 10, 100]
    dataset = args.dataset

    for K in K_values:
        results = []
        for seed in range(3):
            config = get_config(
                algorithm="fedbuff", dataset=dataset,
                concurrency=1000, buffer_size=K,
                seed=seed,
                device=args.device,
                data_dir=args.data_dir,
                log_dir=args.log_dir,
            )
            metrics = run_single(config)
            results.append(metrics)

        trips = [m.trips_to_target for m in results if m.trips_to_target is not None]
        if trips:
            print(f"  K={K:>4}: {np.mean(trips)/1000:.1f}K ± {np.std(trips)/1000:.1f}K")
        else:
            print(f"  K={K:>4}: NOT REACHED")


def run_dp_experiment(args):
    """Reproduce Figure 4: DP experiments on Sent140."""
    print("\n" + "=" * 80)
    print("REPRODUCING FIGURE 4: FedBuff with Differential Privacy")
    print("=" * 80)

    epsilon_values = [6, 12, 24]
    dataset = "sent140"

    for eps in epsilon_values:
        print(f"\n--- ε = {eps} ---")

        # FedBuff + DP-FTRL
        config = get_config(
            algorithm="fedbuff", dataset=dataset,
            concurrency=1000, buffer_size=10,
            seed=0, device=args.device,
            data_dir=args.data_dir, log_dir=args.log_dir,
            dp_enabled=True, dp_mode="ftrl",
        )
        metrics = run_single(config)
        print(f"  FedBuff + DP-FTRL: Final acc = {metrics.final_accuracy:.2f}%")

        # FedAvgM + DP-SGD
        config = get_config(
            algorithm="fedavgm", dataset=dataset,
            concurrency=1000,
            seed=0, device=args.device,
            data_dir=args.data_dir, log_dir=args.log_dir,
            dp_enabled=True, dp_mode="sgd",
        )
        metrics = run_single(config)
        print(f"  FedAvgM + DP-SGD: Final acc = {metrics.final_accuracy:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="FedBuff Reproduction")

    # Experiment mode
    parser.add_argument("--run_table1", action="store_true",
                        help="Reproduce Table 1 (all algorithms comparison)")
    parser.add_argument("--run_scalability", action="store_true",
                        help="Reproduce Figure 3 (scalability)")
    parser.add_argument("--run_buffer_sweep", action="store_true",
                        help="Reproduce Table 2 (effect of K)")
    parser.add_argument("--run_dp", action="store_true",
                        help="Reproduce Figure 4 (DP experiments)")

    # Single experiment params
    parser.add_argument("--algorithm", type=str, default="fedbuff",
                        choices=["fedbuff", "fedasync", "fedavgm", "fedavg", "fedprox"])
    parser.add_argument("--dataset", type=str, default="cifar10",
                        choices=["cifar10", "sent140", "celeba"])
    parser.add_argument("--concurrency", type=int, default=100)
    parser.add_argument("--K", type=int, default=10, help="Buffer size for FedBuff")
    parser.add_argument("--max_trips", type=int, default=600000)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)

    # Hyperparameter overrides
    parser.add_argument("--client_lr", type=float, default=None)
    parser.add_argument("--server_lr", type=float, default=None)
    parser.add_argument("--momentum", type=float, default=None)

    # Infrastructure
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device: cpu or cuda")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--eval_every", type=int, default=5000)

    args = parser.parse_args()

    # Route to experiment mode
    if args.run_table1:
        run_table1(args)
    elif args.run_scalability:
        run_scalability(args)
    elif args.run_buffer_sweep:
        run_buffer_sweep(args)
    elif args.run_dp:
        run_dp_experiment(args)
    else:
        # Single experiment
        overrides = {}
        if args.client_lr is not None:
            overrides["client_lr"] = args.client_lr
        if args.server_lr is not None:
            overrides["server_lr"] = args.server_lr
        if args.momentum is not None:
            overrides["momentum"] = args.momentum

        config = get_config(
            algorithm=args.algorithm,
            dataset=args.dataset,
            concurrency=args.concurrency,
            buffer_size=args.K,
            seed=args.seed,
            device=args.device,
            data_dir=args.data_dir,
            log_dir=args.log_dir,
            max_client_trips=args.max_trips,
            eval_every_trips=args.eval_every,
            num_seeds=args.seeds,
            **overrides,
        )

        if args.seeds > 1:
            run_multi_seed(config)
        else:
            run_single(config)


if __name__ == "__main__":
    main()
