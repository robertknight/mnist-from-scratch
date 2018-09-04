"""
MNIST handwritten digit classifier neural net.
"""

import random
import numpy as np

from loader import load_mnist_images, load_mnist_labels

"""
Just-above-zero value used to avoid using zero in places where that would be undefined.
"""
EPSILON = 1e-7


def onehot(x, length):
    xs = np.zeros(length)
    xs[x] = 1.
    return xs


def shuffle_examples(data, labels):
    shuffled = list(zip(data, labels))
    random.shuffle(shuffled)
    return list(zip(*shuffled))  # unzip


class CategoricalCrossentropy:
    def __call__(self, targets, predictions):
        targets = np.clip(targets, EPSILON, 1.0)
        predictions = np.clip(predictions, EPSILON, 1.0)
        return -np.sum(targets * np.log(predictions))

    def gradient(self, targets, predictions):
        targets = np.clip(targets, EPSILON, 1.0)
        predictions = np.clip(predictions, EPSILON, 1.0)
        return - (targets / predictions)


class Relu:
    """Rectified Linear Unit non-linearity for unit activations."""

    def __call__(self, x):
        return np.maximum(0., x)

    def gradient(self, x):
        return np.diag(np.where(x >= 0., 1., 0.))


class Softmax:
    """Softmax non-linearity for unit activations."""

    def __call__(self, x):
        # Reduce values to avoid overflow.
        # See https://stats.stackexchange.com/a/304774
        s = x - np.max(x)
        exp_s = np.exp(s)
        return exp_s / np.sum(exp_s)

    def gradient(self, x):
        # See https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
        # and https://eli.thegreenplace.net/2016/the-chain-rule-of-calculus/.
        s = self(x)

        out = np.zeros((len(x), len(x)))
        for i in range(0, len(x)):
            for j in range(0, len(x)):
                if i == j:
                    out[i][j] = s[i] * (1 - s[j])
                else:
                    out[i][j] = -s[j] * s[i]
        return out


class Layer:
    """
    A single dense layer in a neural network implementing `y = activation(x * weight)`.
    """
    def __init__(self, unit_count, activation, input_size=None, name=None):
        self.activation = activation
        self.unit_count = unit_count
        self.weights = None
        self.input_size = input_size
        self.name = name

    def connect(self, prev_layer):
        self.input_size = prev_layer.unit_count

    def init_weights(self):
        assert self.input_size is not None
        self.weights = np.random.uniform(-0.2, 0.2, (self.unit_count, self.input_size))

    def forwards(self, inputs):
        z = np.dot(self.weights, inputs)
        return self.activation(z)

    def backwards(self, inputs, loss_grad):
        """
        :param inputs: The inputs to this layer corresponding to `loss_grad`.
        :param loss_grad: Gradient of loss wrt. each of this layer's outputs
        :return: 2-tuple of gradient of inputs and weights wrt. loss.
        """
        z = np.dot(self.weights, inputs)
        z_grad = np.matmul(self.activation.gradient(z), loss_grad)
        weight_grad = np.matmul(z_grad.reshape((self.unit_count, 1)),
                                inputs.reshape((1, self.input_size)))
        input_grad = np.matmul(np.transpose(self.weights), z_grad)

        return (input_grad, weight_grad)


class Model:
    """
    Simple neural network model consisting of a stack of layers.
    """

    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        if len(self.layers) > 0:
            layer.connect(self.layers[-1])
        self.layers.append(layer)

    def fit(self, data, labels, batch_size, epochs, learning_rate, loss_op):
        """Learn parameters given input training `data` and target `labels`."""

        data, labels = shuffle_examples(data, labels)

        # Reset model.
        for layer in self.layers:
            layer.init_weights()

        # Divide training set into mini-batches.
        max_label = np.max(labels)
        batches = []
        for offset in range(0, len(data), batch_size):
            batches.append((data[offset:offset + batch_size], labels[offset:offset + batch_size]))

        for epoch in range(0, epochs):
            epoch_errors = 0
            for batch_index, (batch_data, batch_labels) in enumerate(batches):
                shuffled_batch = list(zip(batch_data, batch_labels))
                np.random.shuffle(shuffled_batch)
                batch_errors = self._fit_batch(shuffled_batch, max_label, loss_op, learning_rate)
                epoch_errors += batch_errors
            epoch_accuracy = 1 - epoch_errors / len(data)
            print(f'epoch {epoch} training accuracy {epoch_accuracy:.3f}...')

    def _fit_batch(self, batch, max_label, loss_op, learning_rate):
        # Compute sum of cost gradients across all examples in batch.
        sum_weight_grads = {}

        for layer in self.layers:
            sum_weight_grads[layer] = np.zeros(layer.weights.shape)

        total_errors = 0

        for i, (example, label) in enumerate(batch):
            target = onehot(label, max_label + 1)
            output = example
            layer_inputs = {}

            for layer in self.layers:
                layer_inputs[layer] = output
                output = layer.forwards(output)

            predicted = np.argmax(output)
            if predicted != label:
                total_errors += 1

            input_grad = loss_op.gradient(target, output)
            for layer in reversed(self.layers):
                inputs = layer_inputs[layer]
                input_grad, weight_grad = layer.backwards(inputs, input_grad)

                sum_weight_grads[layer] += weight_grad

        for layer, sum_weight_grad in sum_weight_grads.items():
            mean_grad = sum_weight_grad / len(batch)
            layer.weights = layer.weights - learning_rate * mean_grad

        return total_errors

    def predict(self, features):
        output = features
        for layer in self.layers:
            output = layer.forwards(output)
        return np.argmax(output)

    def evaluate(self, data, target_labels):
        """
        Evaluate model on test data.

        Return the accuracy (proportion of correctly classified examples).
        """
        errors = 0
        for i, features in enumerate(data):
            predicted_class = self.predict(features)
            actual_class = target_labels[i]
            if predicted_class != actual_class:
                errors += 1
        return 1 - (errors / len(data))


def train_and_test():
    # Debugging.
    np.seterr(divide='raise')

    # Load train and test datasets.
    # These are in the original form provided by http://yann.lecun.com/exdb/mnist/
    print('reading training data...')
    train_images = load_mnist_images('data/train-images.idx3-ubyte')
    train_images = train_images.reshape((60000, 28 * 28))
    train_images = train_images.astype('float') / 255.0
    train_labels = load_mnist_labels('data/train-labels.idx1-ubyte')

    print('reading test data...')
    test_images = load_mnist_images('data/t10k-images.idx3-ubyte')
    test_images = test_images.reshape((10000, 28 * 28))
    test_images = test_images.astype('float') / 255.0
    test_labels = load_mnist_labels('data/t10k-labels.idx1-ubyte')

    model = Model()
    model.add_layer(Layer(10, name='softmax', activation=Softmax(), input_size=28 * 28))

    print('training model...')
    model.fit(train_images, train_labels,
              batch_size=32, epochs=5, learning_rate=0.02,
              loss_op=CategoricalCrossentropy())

    print('evaluating model...')
    accuracy = model.evaluate(test_images, test_labels)
    print('accuracy {}'.format(accuracy))


if __name__ == '__main__':
    train_and_test()
