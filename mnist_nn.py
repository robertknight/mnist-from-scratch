"""
MNIST handwritten digit classifier neural net.
"""

import random
import time
import sys

import numpy as np

from loader import load_mnist_dataset

"""
Just-above-zero value used to avoid zero in places where that would cause undefined results.
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
        shifted_x = x - np.max(x)
        exp_s = np.exp(shifted_x)
        return exp_s / np.sum(exp_s)

    def gradient(self, x):
        # See https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
        # and https://eli.thegreenplace.net/2016/the-chain-rule-of-calculus/.

        softmax_row = self(x)
        softmax_col = softmax_row.reshape((len(x), 1))

        # Compute non-diagonal entries of the jacobian.
        out = -softmax_col * softmax_row

        # Compute diagonal entries of the jacobian.
        out_diag = softmax_row * (1 - softmax_col)

        # Overlay diagonal entries on top of other entries.
        np.copyto(out, out_diag, where=np.eye(len(x), dtype='bool'))

        return out


class ProgressReporter:
    def report_training_progress(self, epoch, total_examples, examples_processed,
                                 epoch_start_time, epoch_total_errors, is_last_batch):
        now = time.time()
        time_per_example = (now - epoch_start_time) / examples_processed
        time_per_example_us = time_per_example * 1000_000
        accuracy = 1 - (epoch_total_errors / examples_processed)
        print('\r', end='')
        print(f'epoch {epoch} ({examples_processed} / {total_examples})  '
              f'{time_per_example_us:.2f}us/example',
              f'  accuracy {accuracy:.3f}',
              end='')
        if is_last_batch:
            print('\n', end='')


class Layer:
    """
    A single dense layer in a neural network implementing `y = activation(x * weight)`.
    """
    def __init__(self, unit_count, activation, input_size=None, name=None):
        self.activation = activation
        self.unit_count = unit_count
        self.weights = None
        self.biases = None
        self.input_size = input_size
        self.name = name

    def init_weights(self):
        assert self.input_size is not None
        self.weights = np.random.uniform(-0.2, 0.2, (self.unit_count, self.input_size))
        self.biases = np.zeros(self.unit_count)

    def forwards(self, inputs):
        z = np.dot(self.weights, inputs) + self.biases
        return self.activation(z)

    def backwards(self, inputs, loss_grad, compute_input_grad=True):
        """
        Compute the gradients with respect to the loss against a training example.

        :param inputs: The inputs to this layer corresponding to `loss_grad`.
        :param loss_grad: Gradient of loss wrt. each of this layer's outputs
        :return: 2-tuple of gradient of inputs and weights wrt. loss.
        """
        z = np.dot(self.weights, inputs) + self.biases
        z_grad = np.matmul(self.activation.gradient(z), loss_grad)
        bias_grad = z_grad
        weight_grad = np.matmul(z_grad.reshape((self.unit_count, 1)),
                                inputs.reshape((1, self.input_size)))

        if compute_input_grad:
            input_grad = np.matmul(np.transpose(self.weights), z_grad)
        else:
            input_grad = None

        return (input_grad, weight_grad, bias_grad)


class Model:
    """
    Simple neural network model consisting of a stack of layers.
    """

    def __init__(self, layers, input_size, progress_reporter=ProgressReporter()):
        self.layers = layers
        self.progress_reporter = progress_reporter

        for i in range(0, len(layers)):
            if i == 0:
                layers[0].input_size = input_size
            else:
                layers[i].input_size = layers[i - 1].unit_count

    def fit(self, data, labels, batch_size, epochs, learning_rate, loss_op,
            learning_rate_decay=0.):

        """Learn parameters given input training `data` and target `labels`."""

        data, labels = shuffle_examples(data, labels)
        reporter = self.progress_reporter

        # Reset model.
        for layer in self.layers:
            layer.init_weights()

        # Divide training set into mini-batches.
        max_label = np.max(labels)
        batches = []
        for offset in range(0, len(data), batch_size):
            batches.append((data[offset:offset + batch_size], labels[offset:offset + batch_size]))

        # Train model.
        for epoch in range(0, epochs):
            epoch_errors = 0
            epoch_start_time = time.time()
            for batch_index, (batch_data, batch_labels) in enumerate(batches):
                shuffled_batch = list(zip(batch_data, batch_labels))
                np.random.shuffle(shuffled_batch)
                batch_errors = self._fit_batch(shuffled_batch, max_label, loss_op, learning_rate)
                epoch_errors += batch_errors

                reporter.report_training_progress(epoch=epoch,
                                                  total_examples=len(data),
                                                  examples_processed=(batch_index + 1) * batch_size,
                                                  epoch_start_time=epoch_start_time,
                                                  epoch_total_errors=epoch_errors,
                                                  is_last_batch=batch_index == len(batches) - 1)
            learning_rate *= (1 - learning_rate_decay)

    def _fit_batch(self, batch, max_label, loss_op, learning_rate):
        # Compute sum of cost gradients across all examples in batch.
        sum_weight_grads = {}
        sum_bias_grads = {}

        for layer in self.layers:
            sum_weight_grads[layer] = np.zeros(layer.weights.shape)
            sum_bias_grads[layer] = np.zeros(layer.biases.shape)

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
                compute_input_grad = layer != self.layers[0]
                input_grad, weight_grad, bias_grad = layer.backwards(inputs, input_grad,
                                                                     compute_input_grad=compute_input_grad)

                sum_weight_grads[layer] += weight_grad
                sum_bias_grads[layer] += bias_grad

        for layer, sum_weight_grad in sum_weight_grads.items():
            mean_grad = sum_weight_grad / len(batch)
            layer.weights = layer.weights - learning_rate * mean_grad

        for layer, sum_bias_grad in sum_bias_grads.items():
            mean_grad = sum_bias_grad / len(batch)
            layer.biases = layer.biases - learning_rate * mean_grad

        return total_errors

    def predict(self, features):
        """
        Predict the class of an example given a feature vector.

        Returns the predicted class label.
        """
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


def train_and_test(dataset_path):
    # Debugging.
    np.seterr(divide='raise')

    # Load train and test datasets.
    print('reading data...')
    train_images, train_labels, test_images, test_labels = load_mnist_dataset(dataset_path)

    model = Model(layers=[
        Layer(32, name='relu', activation=Relu()),
        Layer(10, name='softmax', activation=Softmax()),
    ], input_size=28 * 28)

    print('training model...')
    model.fit(train_images, train_labels,
              batch_size=32, epochs=10, learning_rate=0.1,
              learning_rate_decay=0.1,
              loss_op=CategoricalCrossentropy())

    print('evaluating model...')
    accuracy = model.evaluate(test_images, test_labels)
    print('accuracy {:3f}'.format(accuracy))


if __name__ == '__main__':
    dataset = sys.argv[1]
    train_and_test(dataset)
