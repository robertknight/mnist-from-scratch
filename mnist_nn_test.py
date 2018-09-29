"""
Incomplete tests for a few parts of the neural net.

These were added for debugging purposes and are clearly not comprehensive.
"""

import numpy as np
from numpy.testing import assert_array_equal, assert_almost_equal
import pytest

from mnist_nn import (
    CategoricalCrossentropy,
    FlattenLayer,
    Layer,
    Linear,
    Conv2DLayer,
    Relu,
    Softmax,
    conv2d,
    onehot,
)


class TestOneHot:
    def test_returns_onehot_vec(self):
        assert_array_equal(onehot(2, 5), [0., 0., 1., 0., 0.])

        length = 10
        for i in range(0, length):
            vec = onehot(i, length)
            for k in range(0, length):
                assert vec[k] == (1. if i == k else 0.)


class TestCategoricalCrossentropy:
    @pytest.mark.parametrize('x, y, expected_cc', [
        ([0.1, 0.1, 0.8], [0.8, 0.1, 0.1], 2.0946),
    ])
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
        x = np.array([1, 3, 5])
        y = softmax(x)
        g = softmax.gradient(x)

        for i in range(len(x)):
            dx = np.zeros(len(x))
            dx[i] = 0.1
            xp = x + dx
            yp = softmax(xp)
            assert_almost_equal(yp[i], y[i] + (g[i][i] * dx[i]), 3)


class TestLayer:
    def test_forwards_computes_output(self):
        activation = Relu()
        layer = Layer(1, activation, (2,))
        layer.init_weights()

        inputs = np.array([1, 2])
        output = layer.forwards(inputs)

        assert_almost_equal(output, activation(np.dot(layer.weights, inputs)))

    def test_backwards_computes_gradient(self):
        activation = Relu()
        layer = Layer(2, activation, (2,))
        layer.init_weights()

        # Dummy loss which should drive the weights to zero.
        def dummy_loss(output):
            return np.ones(output.shape)

        inputs = np.array([1., 2.])

        for _ in range(0, 100):
            output = layer.forwards(inputs)
            _, weight_grad, _ = layer.backwards(inputs, dummy_loss(output))
            layer.weights = layer.weights - weight_grad * 0.01

        assert_almost_equal(output, 0.)


class TestConv2d:
    @pytest.mark.parametrize('input_shape,filter_shape', [
        ((5, 5), (3, 3)),
        ((10, 10), (3, 3)),
        ((20, 20), (5, 5)),
    ])
    def test_output_matches_explicit_loops(self, input_shape, filter_shape):
        input_ = np.random.random_sample(input_shape)
        filter_ = np.random.random_sample(filter_shape)

        expected_output = np.zeros(np.subtract(input_shape, filter_shape) + 1)
        for i in range(expected_output.shape[0]):
            for j in range(expected_output.shape[1]):
                input_window = input_[i:i + filter_.shape[0], j:j + filter_.shape[1]]
                expected_output[i][j] = np.sum(np.multiply(input_window, filter_))

        assert_almost_equal(conv2d(input_, filter_), expected_output)


def _minimize_output(layer, input_, steps, learning_rate, train_weights=True):
    """
    Adjust weights or inputs of a Conv2DLayer to minimize abs sum of output.

    This is designed to test that the gradients during backprop push in the
    right direction.

    Returns the losses after each step.
    """

    losses = []
    channels = layer.weights.shape[0]

    for step in range(steps):
        output = layer.forwards(input_)
        loss = abs(np.sum(output))
        losses.append(loss)
        loss_grad = np.sign(output)

        input_grad, weight_grad, _ = layer.backwards(input_, loss_grad, compute_input_grad=False)
        assert weight_grad.shape == layer.weights.shape

        for channel in range(channels):
            if train_weights:
                layer.weights[channel] -= learning_rate * weight_grad[channel]
            else:
                input_ -= learning_rate * input_grad

        learning_rate = learning_rate * 0.9

    return losses


class TestConv2DLayer:

    def test_output_size_is_correct(self):
        input_size = (28, 28)
        layer = Conv2DLayer(64, (3, 3), activation=Relu(), input_size=input_size)
        layer.init_weights()

        assert layer.output_size == (64, 26, 26)

    def test_forwards_returns_correct_output_shape(self):
        input_size = (28, 28)
        layer = Conv2DLayer(64, (3, 3), activation=Relu(), input_size=input_size)
        layer.init_weights()
        input_ = np.random.random_sample(input_size)

        output = layer.forwards(input_)

        assert output.shape == (64, 26, 26)

    def test_forwards_returns_convolution(self):
        input_size = (28, 28)
        layer = Conv2DLayer(64, (3, 3), activation=Linear(), input_size=input_size)
        layer.init_weights()
        input_ = np.random.random_sample(input_size)

        output = layer.forwards(input_)

        for channel in range(64):
            expected_output = conv2d(input_, layer.weights[channel])
            assert_almost_equal(output[channel], expected_output)

    def test_backwards_returns_weight_grad(self):
        input_size = (28, 28)
        layer = Conv2DLayer(64, (3, 3), activation=Linear(), input_size=input_size)
        layer.init_weights()

        # FIXME: If the initial loss happens to be small, the learning rate may
        # be too high and the test can fail.
        input_ = np.random.random_sample(input_size)

        losses = _minimize_output(layer, input_, steps=10, learning_rate=0.001,
                                  train_weights=True)

        assert losses[-1] < losses[0]

    def test_backwards_returns_input_grad(self):
        input_size = (28, 28)
        layer = Conv2DLayer(64, (3, 3), activation=Linear(), input_size=input_size)
        layer.init_weights()

        # FIXME: If the initial loss happens to be small, the learning rate may
        # be too high and the test can fail.
        input_ = np.random.random_sample(input_size) * 10

        losses = _minimize_output(layer, input_, steps=10, learning_rate=0.01,
                                  train_weights=False)

        assert losses[-1] < losses[0]


class TestFlattenLayer:
    def test_output_size_is_correct(self):
        layer = FlattenLayer(input_size=(26, 26, 3))
        assert layer.output_size == (26 * 26 * 3,)

    def test_forwards_returns_flat_output(self):
        input_size = (26, 26, 3)
        layer = FlattenLayer(input_size=input_size)
        input_ = np.random.random_sample(input_size)

        assert layer.forwards(input_).shape == (26 * 26 * 3,)

    def test_backwards_rehsapes_grad(self):
        input_size = (26, 26, 3)
        layer = FlattenLayer(input_size=input_size)
        input_ = np.random.random_sample(input_size)
        loss_grad = np.random.random_sample(layer.output_size)

        assert layer.backwards(input_, loss_grad).shape == input_.shape
