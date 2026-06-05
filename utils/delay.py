"""
Delay distribution simulation for asynchronous FL.

The paper uses half-normal as the default (matches production FL system),
and also evaluates uniform and exponential distributions (Appendix C.1).
"""

import numpy as np
from typing import Literal


class DelaySimulator:
    """Simulates client training delays for asynchronous FL."""

    def __init__(self, distribution: Literal["half_normal", "uniform", "exponential"] = "half_normal",
                 scale: float = 10.0, seed: int = 0):
        """
        Args:
            distribution: Delay distribution type.
            scale: Scale parameter controlling delay magnitude.
            seed: Random seed for reproducibility.
        """
        self.distribution = distribution
        self.scale = scale
        self.rng = np.random.RandomState(seed)

    def sample(self, n: int = 1) -> np.ndarray:
        """Sample n delay values (in arbitrary time units).
        
        Returns:
            Array of positive delay values.
        """
        if self.distribution == "half_normal":
            # |N(0, scale)| — matches production FL system (Figure 5)
            return np.abs(self.rng.normal(0, self.scale, size=n))
        elif self.distribution == "uniform":
            return self.rng.uniform(0, 2 * self.scale, size=n)
        elif self.distribution == "exponential":
            return self.rng.exponential(self.scale, size=n)
        else:
            raise ValueError(f"Unknown distribution: {self.distribution}")

    def sample_one(self) -> float:
        return self.sample(1)[0]


class StalenessTracker:
    """Tracks staleness of client updates in async FL.
    
    Staleness τ_i(t) = number of server updates between when
    client i downloaded the model and when its update is used.
    """

    def __init__(self):
        self.current_server_version = 0

    def record_download(self) -> int:
        """Record that a client downloaded the current model version."""
        return self.current_server_version

    def compute_staleness(self, download_version: int) -> int:
        """Compute staleness given the version the client started from."""
        return self.current_server_version - download_version

    def server_step(self):
        """Increment server version after an update."""
        self.current_server_version += 1


def staleness_scaling(staleness: int, exponent: float = 0.5) -> float:
    """Down-weight stale updates: s(τ) = 1 / (1 + τ)^exponent.
    
    From Section 5, similar to Xie et al. (2019).
    Default exponent=0.5 as used in experiments.
    """
    return 1.0 / (1.0 + staleness) ** exponent
