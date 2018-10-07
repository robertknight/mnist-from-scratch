import os

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def seed_rng():
    """Seed numpy's PRNG to make reproducing test failures easier."""
    seed = os.getenv("RANDOM_SEED") or np.random.randint(1e6)
    seed = int(seed)
    print(f"numpy random seed {seed}. Set RANDOM_SEED env var to reproduce.")
    np.random.seed(42)
