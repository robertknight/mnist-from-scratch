import numpy as np
from numpy.testing import assert_array_equal, assert_almost_equal
import pytest

from mnist import (
    CategoricalCrossentropy,
    Layer,
    Model,
    Relu,
    Softmax,
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
            assert_almost_equal(yp[i], y[i] + (g[i] * dx[i]), 3)


class TestLayer:
    def test_forwards_computes_output(self):
        activation = Relu()
        layer = Layer(1, activation, 2)
        layer.init_weights()

        inputs = np.array([1, 2])
        output = layer.forwards(inputs)

        assert_almost_equal(output, activation(np.dot(layer.weights, inputs)))

    def test_backwards_computes_gradient(self):
        activation = Relu()
        layer = Layer(2, activation, 2)
        layer.init_weights()

        # Dummy loss which should drive the weights to zero.
        def dummy_loss(output):
            return np.ones(output.shape)

        inputs = np.array([1., 2.])

        for _ in range(0, 100):
            output = layer.forwards(inputs)
            _, weight_grad = layer.backwards(inputs, dummy_loss(output))
            layer.weights = layer.weights - weight_grad * 0.01

        assert_almost_equal(output, 0.)


class TestModel:
    def test_it_learns_xor(self):
        data = [([0., 0.], 0),
                ([0., 1.], 1),
                ([1., 0.], 1),
                ([1., 1.], 0),
                ]
        train_data, train_labels = zip(*data)
        repeats = 1000
        train_data = [np.array(d) for d in train_data * repeats]
        train_labels = train_labels * repeats

        model = Model()
        model.add_layer(Layer(2, name='relu', activation=Relu(), input_size=2))
        model.add_layer(Layer(2, name='output', activation=Relu()))

        model.fit(train_data, train_labels, batch_size=2, epochs=10, learning_rate=0.02,
                  loss_op=CategoricalCrossentropy())

        accuracy = model.evaluate(train_data, train_labels)

        assert accuracy == 1.0
