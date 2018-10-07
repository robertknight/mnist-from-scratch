import numpy as np
import pytest

from neuralnet.layers import Layer
from neuralnet.model import Model
from neuralnet.ops import CategoricalCrossentropy, Relu, Softmax


class TestModel:
    CLASS_COUNT = 3

    def test_fit_trains_model(self, simple_model, generate_batch):
        train_data, train_labels = generate_batch(2000)
        test_data, test_labels = generate_batch(100)

        simple_model.fit(
            train_data,
            train_labels,
            batch_size=10,
            epochs=3,
            learning_rate=0.1,
            loss_op=CategoricalCrossentropy(),
        )

        accuracy = simple_model.evaluate(test_data, test_labels)

        assert accuracy > 0.90

    @pytest.fixture
    def simple_model(self, generate_data):
        x = generate_data(0)
        layers = [
            Layer(5, activation=Relu()),
            Layer(self.CLASS_COUNT, activation=Softmax()),
        ]
        return Model(layers, input_size=x.shape)

    @pytest.fixture
    def generate_batch(self, generate_data):
        def gen(count):
            data = []
            labels = []
            for _ in range(count):
                label = np.random.randint(0, self.CLASS_COUNT)
                example = generate_data(label)
                data.append(example)
                labels.append(label)
            return data, labels

        return gen

    @pytest.fixture
    def generate_data(self):
        def gen(mean):
            # Generate an easily classified data sample.
            return np.random.normal(mean, 0.1, (5,))

        return gen
