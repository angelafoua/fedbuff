"""
Model architectures used in the FedBuff paper.
- Sent140: 2-layer LSTM binary classifier (GloVe embeddings)
- CelebA: 4-layer CNN with GroupNorm
- CIFAR-10: 4-layer CNN with GroupNorm
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMClassifier(nn.Module):
    """2-layer LSTM for Sent140 binary sentiment classification.
    
    Architecture from LEAF benchmark:
    - 300D GloVe embedding (top 10K vocab, frozen)
    - 2 LSTM layers, 100 hidden units
    - 128-unit linear layer
    - Binary output
    """

    def __init__(self, vocab_size=10000, embedding_dim=300, hidden_dim=100,
                 output_dim=2, num_layers=2, dropout=0.1, pretrained_embeddings=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            self.embedding.weight.requires_grad = False

        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.fc1 = nn.Linear(hidden_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, seq_len) integer token IDs
        embedded = self.embedding(x)  # (batch, seq_len, 300)
        _, (hidden, _) = self.lstm(embedded)
        hidden = self.dropout(hidden[-1])  # last layer hidden state
        out = F.relu(self.fc1(hidden))
        out = self.fc2(out)
        return out


class CelebACNN(nn.Module):
    """4-layer CNN for CelebA binary classification.
    
    Uses GroupNorm instead of BatchNorm (Hsieh et al., 2020).
    Input: 32x32 RGB images.
    """

    def __init__(self, num_classes=2, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, stride=1, padding=2)
        self.gn1 = nn.GroupNorm(8, 32)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=2)
        self.gn2 = nn.GroupNorm(8, 64)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=2)
        self.gn3 = nn.GroupNorm(8, 128)
        self.conv4 = nn.Conv2d(128, 256, 3, stride=1, padding=2)
        self.gn4 = nn.GroupNorm(8, 256)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout)

        # Calculate flattened size after 4 conv+pool layers on 32x32 input
        # Each conv with padding=2 on 3x3 kernel adds 2 to each dim, then pool halves
        # 32 -> 34 -> 17 -> 19 -> 9 -> 11 -> 5 -> 7 -> 3
        self._flat_size = self._get_flat_size()
        self.fc = nn.Linear(self._flat_size, num_classes)

    def _get_flat_size(self):
        x = torch.zeros(1, 3, 32, 32)
        x = self.pool(F.relu(self.gn1(self.conv1(x))))
        x = self.pool(F.relu(self.gn2(self.conv2(x))))
        x = self.pool(F.relu(self.gn3(self.conv3(x))))
        x = self.pool(F.relu(self.gn4(self.conv4(x))))
        return x.view(1, -1).size(1)

    def forward(self, x):
        x = self.pool(F.relu(self.gn1(self.conv1(x))))
        x = self.pool(F.relu(self.gn2(self.conv2(x))))
        x = self.pool(F.relu(self.gn3(self.conv3(x))))
        x = self.pool(F.relu(self.gn4(self.conv4(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class CIFAR10CNN(nn.Module):
    """4-layer CNN for CIFAR-10 multi-class classification.
    
    Same architecture as CelebA but with 10-class output and GroupNorm.
    """

    def __init__(self, num_classes=10, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, stride=1, padding=2)
        self.gn1 = nn.GroupNorm(8, 32)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=2)
        self.gn2 = nn.GroupNorm(8, 64)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=2)
        self.gn3 = nn.GroupNorm(8, 128)
        self.conv4 = nn.Conv2d(128, 256, 3, stride=1, padding=2)
        self.gn4 = nn.GroupNorm(8, 256)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout)

        self._flat_size = self._get_flat_size()
        self.fc = nn.Linear(self._flat_size, num_classes)

    def _get_flat_size(self):
        x = torch.zeros(1, 3, 32, 32)
        x = self.pool(F.relu(self.gn1(self.conv1(x))))
        x = self.pool(F.relu(self.gn2(self.conv2(x))))
        x = self.pool(F.relu(self.gn3(self.conv3(x))))
        x = self.pool(F.relu(self.gn4(self.conv4(x))))
        return x.view(1, -1).size(1)

    def forward(self, x):
        x = self.pool(F.relu(self.gn1(self.conv1(x))))
        x = self.pool(F.relu(self.gn2(self.conv2(x))))
        x = self.pool(F.relu(self.gn3(self.conv3(x))))
        x = self.pool(F.relu(self.gn4(self.conv4(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


def get_model(dataset_name: str, **kwargs) -> nn.Module:
    """Factory for creating the appropriate model for each dataset."""
    if dataset_name == "sent140":
        return LSTMClassifier(**kwargs)
    elif dataset_name == "celeba":
        return CelebACNN(**kwargs)
    elif dataset_name == "cifar10":
        return CIFAR10CNN(**kwargs)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
