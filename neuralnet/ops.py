import numpy as np

from .util import EPSILON


class CategoricalCrossentropy:
    def __call__(self, targets, predictions):
        targets = np.clip(targets, EPSILON, 1.0)
        predictions = np.clip(predictions, EPSILON, 1.0)
        return -np.sum(targets * np.log(predictions))

    def gradient(self, targets, predictions):
        targets = np.clip(targets, EPSILON, 1.0)
        predictions = np.clip(predictions, EPSILON, 1.0)
        return -(targets / predictions)


class Linear:
    """Linear activation."""

    def __call__(self, x):
        return x

    def gradient(self, x, grads):
        return x


class Relu:
    """Rectified Linear Unit non-linearity for unit activations."""

    def __call__(self, x):
        return np.maximum(0.0, x)

    def gradient(self, x, grads):
        return (
            np.where(
                x >= 0.0, np.array(1.0, dtype="float32"), np.array(0.0, dtype="float32")
            )
            * grads
        )


class Softmax:
    """Softmax non-linearity for unit activations."""

    def __call__(self, x):
        # Reduce values to avoid overflow.
        # See https://stats.stackexchange.com/a/304774
        shifted_x = x - np.max(x)
        exp_s = np.exp(shifted_x)
        return exp_s / np.sum(exp_s)

    def gradient(self, x, grads):
        # Adapted from `_SoftmaxGradient` in Tensorflow's `nn_grad.py`.
        softmax = self(x)
        return (softmax * grads) - np.sum(softmax * grads, -1, keepdims=True) * softmax
