import numpy as np
from numpy.testing import assert_almost_equal

import pytest

from neuralnet.ops import CategoricalCrossentropy, Softmax


class TestCategoricalCrossentropy:
    @pytest.mark.parametrize(
        "x, y, expected_cc", [([0.1, 0.1, 0.8], [0.8, 0.1, 0.1], 2.0946)]
    )
    def test_call(self, x, y, expected_cc):
        cc = CategoricalCrossentropy()
        x = np.array(x)
        y = np.array(y)
        crossentropy = cc(x, y)
        assert_almost_equal(crossentropy, expected_cc, 3)


class TestSoftmax:
    def test_call(self):
        softmax = Softmax()
        y = softmax(np.array([1, 3, 5]))
        assert_almost_equal(y, [0.0158762, 0.1173104, 0.8668133])

    def test_gradient(self):
        softmax = Softmax()
        loss_op = CategoricalCrossentropy()
        x = np.array([1.0, 3.0, 5.0])
        target = np.array([1.0, 0.01, 0.01])

        losses = []
        for _ in range(5):
            y = softmax(x)
            losses.append(loss_op(target, y))
            loss_grad = loss_op.gradient(target, y)
            g = softmax.gradient(x, loss_grad)
            x -= 0.1 * g

        assert_almost_equal(
            losses, [4.1657903, 3.9919632, 3.8205074, 3.6516005, 3.485433]
        )
