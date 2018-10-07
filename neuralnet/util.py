import numpy as np
import random

"""
Just-above-zero value used to avoid zero in places where that would cause undefined results.
"""
EPSILON = 1e-7


def onehot(x, length):
    xs = np.zeros(length, dtype=np.float32)
    xs[x] = 1.0
    return xs


def shuffle_examples(data, labels):
    shuffled = list(zip(data, labels))
    random.shuffle(shuffled)
    return list(zip(*shuffled))  # unzip
