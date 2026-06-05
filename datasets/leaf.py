"""
LEAF benchmark dataset loaders for Sent140 and CelebA.

These require downloading and preprocessing from the LEAF repository:
https://github.com/TalwalkarLab/leaf

Sent140: 660,120 clients (Twitter accounts), binary sentiment classification
CelebA: 9,343 clients (celebrities), binary face attribute classification
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional, Tuple
from pathlib import Path


class LEAFDataset(Dataset):
    """Generic LEAF dataset for a single client."""

    def __init__(self, x_data, y_data, transform=None):
        self.x = x_data
        self.y = y_data
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        if self.transform:
            x = self.transform(x)
        return x, y


class Sent140FL:
    """Sent140 federated dataset from LEAF benchmark.
    
    Binary sentiment classification on tweets.
    Each Twitter account = one client.
    Uses 2-layer LSTM with GloVe embeddings.
    
    Setup:
        cd leaf/data/sent140
        ./preprocess.sh -s niid --sf 0.15 --tf 0.8 --t sample --k 0
    """

    def __init__(self, data_dir: str = "./data/sent140", max_seq_len: int = 25,
                 vocab_size: int = 10000):
        self.data_dir = data_dir
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.client_data: Dict[str, Dict] = {}
        self.client_ids: List[str] = []

        self._load_data()

    def _load_data(self):
        """Load preprocessed LEAF data from JSON files."""
        train_dir = os.path.join(self.data_dir, "train")
        test_dir = os.path.join(self.data_dir, "test")

        if not os.path.exists(train_dir):
            print(f"WARNING: Sent140 data not found at {train_dir}")
            print("Please download and preprocess using LEAF:")
            print("  git clone https://github.com/TalwalkarLab/leaf")
            print("  cd leaf/data/sent140")
            print("  ./preprocess.sh -s niid --sf 0.15 --tf 0.8 --t sample --k 0")
            self._create_dummy_data()
            return

        # Load training data
        for fname in sorted(os.listdir(train_dir)):
            if not fname.endswith('.json'):
                continue
            with open(os.path.join(train_dir, fname)) as f:
                data = json.load(f)

            for user_id, user_data in zip(data['users'], data['user_data'].values()):
                self.client_data[user_id] = {
                    'train_x': user_data['x'],
                    'train_y': user_data['y'],
                    'test_x': [],
                    'test_y': [],
                }
                self.client_ids.append(user_id)

        # Load test data
        if os.path.exists(test_dir):
            for fname in sorted(os.listdir(test_dir)):
                if not fname.endswith('.json'):
                    continue
                with open(os.path.join(test_dir, fname)) as f:
                    data = json.load(f)

                for user_id, user_data in zip(data['users'], data['user_data'].values()):
                    if user_id in self.client_data:
                        self.client_data[user_id]['test_x'] = user_data['x']
                        self.client_data[user_id]['test_y'] = user_data['y']

    def _create_dummy_data(self):
        """Create minimal dummy data for testing without LEAF download."""
        print("Creating dummy Sent140 data for testing...")
        for i in range(100):
            user_id = f"dummy_user_{i}"
            n_samples = np.random.randint(5, 50)
            self.client_data[user_id] = {
                'train_x': [[np.random.randint(0, self.vocab_size)
                              for _ in range(self.max_seq_len)]
                             for _ in range(n_samples)],
                'train_y': [np.random.randint(0, 2) for _ in range(n_samples)],
                'test_x': [],
                'test_y': [],
            }
            self.client_ids.append(user_id)

    def get_client_dataset(self, client_idx: int) -> LEAFDataset:
        user_id = self.client_ids[client_idx]
        data = self.client_data[user_id]
        x = torch.LongTensor(data['train_x'])
        y = torch.LongTensor(data['train_y'])
        return LEAFDataset(x, y)

    def get_test_loader(self, batch_size: int = 128) -> DataLoader:
        all_x, all_y = [], []
        for user_data in self.client_data.values():
            if user_data['test_x']:
                all_x.extend(user_data['test_x'])
                all_y.extend(user_data['test_y'])
        if not all_x:
            # Use last 10% of training data as test
            for user_data in self.client_data.values():
                n = len(user_data['train_y'])
                split = max(1, int(0.9 * n))
                all_x.extend(user_data['train_x'][split:])
                all_y.extend(user_data['train_y'][split:])

        dataset = LEAFDataset(torch.LongTensor(all_x), torch.LongTensor(all_y))
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)

    def get_val_loader(self, batch_size: int = 128) -> DataLoader:
        return self.get_test_loader(batch_size)

    @property
    def num_actual_clients(self) -> int:
        return len(self.client_ids)


class CelebAFL:
    """CelebA federated dataset from LEAF benchmark.
    
    Binary classification on face attributes (smiling).
    Each celebrity = one client. 9,343 clients.
    Images resized to 32x32, normalized by 0.5 mean/std.
    
    Setup:
        cd leaf/data/celeba
        ./preprocess.sh -s niid --sf 1.0 --tf 0.8 --t sample --k 5
    """

    def __init__(self, data_dir: str = "./data/celeba"):
        self.data_dir = data_dir
        self.client_data: Dict[str, Dict] = {}
        self.client_ids: List[str] = []

        from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        self._load_data()

    def _load_data(self):
        train_dir = os.path.join(self.data_dir, "train")
        if not os.path.exists(train_dir):
            print(f"WARNING: CelebA data not found at {train_dir}")
            print("Please download and preprocess using LEAF:")
            print("  git clone https://github.com/TalwalkarLab/leaf")
            print("  cd leaf/data/celeba")
            print("  ./preprocess.sh -s niid --sf 1.0 --tf 0.8 --t sample --k 5")
            self._create_dummy_data()
            return

        for fname in sorted(os.listdir(train_dir)):
            if not fname.endswith('.json'):
                continue
            with open(os.path.join(train_dir, fname)) as f:
                data = json.load(f)
            for user_id, user_data in zip(data['users'], data['user_data'].values()):
                self.client_data[user_id] = user_data
                self.client_ids.append(user_id)

    def _create_dummy_data(self):
        print("Creating dummy CelebA data for testing...")
        for i in range(100):
            user_id = f"celeb_{i}"
            n = np.random.randint(5, 30)
            self.client_data[user_id] = {
                'x': [np.random.rand(3, 32, 32).tolist() for _ in range(n)],
                'y': [np.random.randint(0, 2) for _ in range(n)],
            }
            self.client_ids.append(user_id)

    def get_client_dataset(self, client_idx: int) -> LEAFDataset:
        user_id = self.client_ids[client_idx]
        data = self.client_data[user_id]
        x = torch.FloatTensor(data['x'])
        y = torch.LongTensor(data['y'])
        return LEAFDataset(x, y)

    def get_test_loader(self, batch_size: int = 128) -> DataLoader:
        all_x, all_y = [], []
        for user_data in self.client_data.values():
            n = len(user_data['y'])
            split = max(1, int(0.9 * n))
            all_x.extend(user_data['x'][split:])
            all_y.extend(user_data['y'][split:])
        dataset = LEAFDataset(torch.FloatTensor(all_x), torch.LongTensor(all_y))
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)

    def get_val_loader(self, batch_size: int = 128) -> DataLoader:
        return self.get_test_loader(batch_size)

    @property
    def num_actual_clients(self) -> int:
        return len(self.client_ids)
