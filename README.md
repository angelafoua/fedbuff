# FedBuff Reproduction

Reproduction of "Federated Learning with Buffered Asynchronous Aggregation" (Nguyen et al., AISTATS 2022).

## Structure

```
fedbuff/
├── algorithms/
│   ├── base.py          # Base FL server/client classes
│   ├── fedbuff.py       # FedBuff (buffered async aggregation)
│   ├── fedavg.py        # FedAvg + FedAvgM + FedProx (synchronous)
│   └── fedasync.py      # FedAsync (fully asynchronous)
├── datasets/
│   ├── sent140.py       # Sent140 tweet sentiment dataset
│   ├── celeba.py        # CelebA face attributes dataset
│   └── cifar10.py       # CIFAR-10 with Dirichlet non-IID split
├── utils/
│   ├── delay.py         # Delay distribution simulation
│   ├── metrics.py       # Evaluation and logging
│   ├── dp.py            # Differential privacy (DP-FTRL, DP-SGD)
│   └── models.py        # LSTM, CNN model architectures
├── config.py            # Experiment configurations
├── simulator.py         # Core FL simulator (async + sync)
├── run_experiment.py    # Main entry point
└── README.md
```

## Quick Start

```bash
pip install torch torchvision numpy scipy tqdm tensorboard --break-system-packages

# Run FedBuff on CIFAR-10
python run_experiment.py --algorithm fedbuff --dataset cifar10 --concurrency 100 --K 10

# Run FedAvgM baseline
python run_experiment.py --algorithm fedavgm --dataset cifar10 --concurrency 100

# Run full comparison (Table 1 reproduction)
python run_experiment.py --run_table1

# Run scalability experiment (Figure 3)
python run_experiment.py --run_scalability
```

## Key Hyperparameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--K` | Buffer size for FedBuff | 10 |
| `--concurrency` | Max clients training in parallel | 1000 |
| `--client_lr` | Client learning rate η_l | tuned per task |
| `--server_lr` | Server learning rate η_g | tuned per task |
| `--momentum` | Server momentum β | 0.9 |
| `--batch_size` | Client batch size B | 32 |
| `--seeds` | Number of seeds to average | 3 |

## Datasets

- **Sent140**: LEAF benchmark Twitter sentiment (660K clients). Auto-downloads.
- **CelebA**: LEAF benchmark face attributes (9.3K clients). Requires manual download.
- **CIFAR-10**: Dirichlet(0.1) non-IID partition into 5000 clients. Auto-downloads via torchvision.
