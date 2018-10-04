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
    MaxPoolingLayer,
    Conv2DLayer,
    Relu,
    Softmax,
    conv2d,
    onehot,
)


class TestOneHot:
    def test_returns_onehot_vec(self):
        assert_array_equal(onehot(2, 5), [0.0, 0.0, 1.0, 0.0, 0.0])

        length = 10
        for i in range(0, length):
            vec = onehot(i, length)
            for k in range(0, length):
                assert vec[k] == (1.0 if i == k else 0.0)


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

        inputs = np.array([1.0, 2.0])

        for _ in range(0, 100):
            output = layer.forwards(inputs)
            _, weight_grad, _ = layer.backwards(dummy_loss(output))
            layer.weights = layer.weights - weight_grad * 0.01

        assert_almost_equal(output, 0.0)


class TestConv2d:
    @pytest.mark.parametrize(
        "input_shape,filter_shape",
        [((5, 5), (3, 3)), ((10, 10), (3, 3)), ((20, 20), (5, 5))],
    )
    def test_single_channel_output_matches_explicit_loops(
        self, input_shape, filter_shape
    ):
        input_ = np.random.random_sample(input_shape)
        filter_ = np.random.random_sample(filter_shape)

        expected_output = np.zeros(np.subtract(input_shape, filter_shape) + 1)
        for i in range(expected_output.shape[0]):
            for j in range(expected_output.shape[1]):
                input_window = input_[
                    i : i + filter_.shape[0], j : j + filter_.shape[1]
                ]
                expected_output[i][j] = np.sum(np.multiply(input_window, filter_))
        expected_output = expected_output.reshape((1, *expected_output.shape))

        input_with_channel = input_.reshape((1, *input_shape))
        filter_with_channel = filter_.reshape((1, 1, *filter_shape))
        assert_almost_equal(
            conv2d(input_with_channel, filter_with_channel), expected_output
        )


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

        layer.forwards(input_)
        input_grad, weight_grad, _ = layer.backwards(
            loss_grad, compute_input_grad=not train_weights
        )
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
        input_size = (1, 28, 28)
        layer = Conv2DLayer(64, (3, 3), activation=Relu(), input_size=input_size)
        layer.init_weights()

        assert layer.output_size == (64, 26, 26)

    def test_forwards_returns_correct_output_shape(self):
        input_size = (1, 28, 28)
        layer = Conv2DLayer(64, (3, 3), activation=Relu(), input_size=input_size)
        layer.init_weights()
        input_ = np.random.random_sample(input_size)

        output = layer.forwards(input_)

        assert output.shape == (64, 26, 26)

    def test_forwards_returns_convolution(self):
        input_size = (1, 28, 28)
        layer = Conv2DLayer(64, (3, 3), activation=Linear(), input_size=input_size)
        layer.init_weights()
        input_ = np.random.random_sample(input_size)

        output = layer.forwards(input_)

        for channel in range(64):
            expected_output = conv2d(input_, layer.weights)
            assert_almost_equal(output, expected_output)

    def test_backwards_returns_weight_grad(self):
        input_size = (1, 28, 28)
        layer = Conv2DLayer(64, (3, 3), activation=Linear(), input_size=input_size)
        layer.init_weights()

        # FIXME: If the initial loss happens to be small, the learning rate may
        # be too high and the test can fail.
        input_ = np.random.random_sample(input_size)

        losses = _minimize_output(
            layer, input_, steps=10, learning_rate=0.001, train_weights=True
        )

        assert losses[-1] < losses[0]

    def test_backwards_returns_input_grad(self):
        input_size = (1, 28, 28)
        layer = Conv2DLayer(64, (3, 3), activation=Linear(), input_size=input_size)
        layer.init_weights()

        # FIXME: If the initial loss happens to be small, the learning rate may
        # be too high and the test can fail.
        input_ = np.random.random_sample(input_size) * 10

        losses = _minimize_output(
            layer, input_, steps=10, learning_rate=0.01, train_weights=False
        )

        assert losses[-1] < losses[0]

    def test_can_stack_layers(self):
        input_size = (1, 28, 28)
        layer1 = Conv2DLayer(64, (3, 3), activation=Linear(), input_size=input_size)
        layer1.init_weights()
        layer2 = Conv2DLayer(
            32, (3, 3), activation=Linear(), input_size=layer1.output_size
        )
        layer2.init_weights()
        input_ = np.random.random_sample(input_size) * 10

        layer1_output = layer1.forwards(input_)
        layer2_output = layer2.forwards(layer1_output)

        assert layer1.weights.shape == (64, 1, 3, 3)
        assert layer2.weights.shape == (32, 64, 3, 3)
        assert layer2_output.shape == (32, 24, 24)


class TestFlattenLayer:
    def test_output_size_is_correct(self):
        layer = FlattenLayer(input_size=(26, 26, 3))
        assert layer.output_size == (26 * 26 * 3,)

    def test_forwards_returns_flat_output(self):
        input_size = (26, 26, 3)
        layer = FlattenLayer(input_size=input_size)
        input_ = np.random.random_sample(input_size)

        assert layer.forwards(input_).shape == (26 * 26 * 3,)

    def test_backwards_reshapes_grad(self):
        input_size = (26, 26, 3)
        layer = FlattenLayer(input_size=input_size)
        input_ = np.random.random_sample(input_size)
        loss_grad = np.random.random_sample(layer.output_size)

        layer.forwards(input_)
        input_grad, *rest = layer.backwards(loss_grad)

        assert input_grad.shape == input_.shape


class TestMaxPoolingLayer:
    def test_output_size_is_correct(self):
        layer = MaxPoolingLayer((2, 2), input_size=(32, 26, 26))
        assert layer.output_size == (32, 13, 13)

    def test_forwards_pools_input(self):
        # nb. `fmt: off` triggers a Black bug here.
        input_ = np.array([[1, 0, 0, 2], [0, 0, 0, 0], [0, 0, 0, 0], [3, 0, 0, 4]])
        input_ = np.stack([input_, input_, input_])
        layer = MaxPoolingLayer((2, 2), input_size=input_.shape)

        pooled = layer.forwards(input_)

        assert pooled.shape == (3, 2, 2)

        for channel in range(input_.shape[0]):
            assert_almost_equal(pooled[channel], [[1., 2], [3, 4]])

    def test_backwards_routes_gradient(self):
        input_ = np.array([[1, 0, 0, 2], [0, 0, 0, 0], [0, 0, 0, 0], [3, 0, 0, 4]])
        input_ = np.stack([input_, input_, input_])
        layer = MaxPoolingLayer((2, 2), input_size=input_.shape)

        layer.forwards(input_)
        loss_grad = [[0.1, 0.2], [0.3, 0.4]]
        loss_grad = np.stack([loss_grad, loss_grad, loss_grad])
        input_grad, *rest = layer.backwards(loss_grad)

        for channel in range(input_.shape[0]):
            expected_grad = [
                [0.1, 0.0, 0.0, 0.2],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.3, 0.0, 0.0, 0.4],
            ]
            assert_almost_equal(input_grad[channel], expected_grad)

    def test_supports_input_not_a_multiple_of_pool_size(self):
        input_ = np.ones((1, 13, 13))
        loss_grad = np.ones((1, 6, 6))
        layer = MaxPoolingLayer((2, 2), input_size=input_.shape)

        output = layer.forwards(input_)
        grad, *rest = layer.backwards(loss_grad)

        assert output.shape == loss_grad.shape
        assert grad.shape == input_.shape
