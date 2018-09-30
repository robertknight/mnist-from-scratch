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


class Linear:
    """Linear activation."""

    def __call__(self, x):
        return x

    def gradient(self, x):
        return np.ones(x.shape)


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

    @property
    def output_size(self):
        return (self.unit_count,)

    def init_weights(self):
        assert self.input_size is not None
        assert len(self.input_size) == 1

        self.weights = np.random.uniform(-0.2, 0.2, (self.unit_count, *self.input_size))
        self.biases = np.zeros(self.unit_count)

    def forwards(self, inputs):
        z = np.dot(self.weights, inputs) + self.biases
        self.last_inputs = inputs
        self.last_z = z
        return self.activation(z)

    def backwards(self, loss_grad, compute_input_grad=True):
        """
        Compute the gradients with respect to the loss against a training example.

        :param loss_grad: Gradient of loss wrt. each of this layer's outputs
        :return: 2-tuple of gradient of inputs and weights wrt. loss.
        """
        z_grad = np.matmul(self.activation.gradient(self.last_z), loss_grad)
        bias_grad = z_grad
        weight_grad = np.matmul(z_grad.reshape((self.unit_count, 1)),
                                self.last_inputs.reshape((1, *self.input_size)))

        if compute_input_grad:
            input_grad = np.matmul(np.transpose(self.weights), z_grad)
        else:
            input_grad = None

        return (input_grad, weight_grad, bias_grad)


def conv2d(matrix, filter_):
    """
    Return a 2D convolution of an input matrix with a filter.

    This implements 2D convolution without requiring nested loops in Python,
    which would be slow for large input matrices.

    Adapted from https://stackoverflow.com/questions/43086557/.
    """

    # Create a view on the input matrix which is a `filter_.shape`-sized grid
    # of 2D windows, where each window contains all the elements that
    # should be multiplied by a given filter element during convolution.
    s = filter_.shape + tuple(np.subtract(matrix.shape, filter_.shape) + 1)
    as_strided = np.lib.stride_tricks.as_strided
    windows = as_strided(matrix, shape=s, strides=matrix.strides * 2)

    # Multiply each window by corresponding filter element, and then sum the
    # corresponding elements of each window.
    return np.einsum('ij,ijkl->kl', filter_, windows)


class Conv2DLayer:
    """
    A 2D convolution layer.
    """

    def __init__(self, channels, filter_shape, activation, input_size=None, name=None):
        self.channels = channels
        self.filter_shape = filter_shape
        self.activation = activation
        self.input_size = input_size
        self.biases = None
        self.name = name

    @property
    def output_size(self):
        output_shape = np.subtract(self.input_size, self.filter_shape) + 1
        return (self.channels, *output_shape)

    def init_weights(self):
        assert self.input_size is not None

        self.biases = None
        self.weights = np.random.uniform(-0.2, 0.2, (self.channels, *self.filter_shape))

    def forwards(self, inputs):
        assert inputs.shape == self.input_size

        channel_outputs = []
        for channel in range(self.channels):
            channel_output = conv2d(inputs, self.weights[channel])
            channel_output = self.activation(channel_output)
            channel_outputs.append(channel_output)
        self.last_outputs = np.stack(channel_outputs, axis=0)
        self.last_inputs = inputs

        return self.last_outputs

    def backwards(self, loss_grad, compute_input_grad=True):
        assert loss_grad.shape == self.output_size

        filter_w, filter_h = self.filter_shape
        input_w, input_h = self.input_size

        bias_grad = None  # Not implemented yet.

        output_shape = self.output_size

        # Compute gradients of activation input wrt. loss.
        activation_grads = np.zeros(output_shape)
        for channel in range(self.channels):
            channel_output = self.last_outputs[channel]
            activation_grads[channel] = np.matmul(
                self.activation.gradient(channel_output), loss_grad[channel]
            )

        # Compute gradient of weights wrt. loss.
        weight_grad = np.zeros(self.weights.shape)

        # Compute weight gradient for each element of filter.
        # The filter is typically much smaller than the input so we get
        # more efficient vectorization than looping over windows in the input.
        for y in range(filter_h):
            for x in range(filter_w):
                filter_inputs = self.last_inputs[
                    y:input_h - filter_h + y + 1,
                    x:input_w - filter_w + x + 1
                ]
                weight_grad[:, y, x] = np.einsum('ij,Cij->C', filter_inputs, activation_grads)
                weight_grad[:, y, x] /= np.product(filter_inputs.shape)

        if compute_input_grad:
            # Compute gradient of inputs wrt. loss.
            input_grad = np.zeros(self.input_size)

            # Number of times each input position was multiplied by a weight.
            weight_counts = np.zeros(self.input_size)

            weight_grads = np.einsum('Cyx,Cij->yxij', self.weights, activation_grads)
            for y in range(filter_h):
                for x in range(filter_w):
                    input_grad_window = input_grad[
                        y:input_h - filter_h + y + 1,
                        x:input_w - filter_w + x + 1
                    ]
                    np.copyto(input_grad_window, weight_grads[y, x])

                    weight_count_window = weight_counts[
                        y:input_h - filter_h + y + 1,
                        x:input_w - filter_w + x + 1
                    ]
                    weight_count_window += np.ones(weight_count_window.shape) * self.channels
            input_grad /= weight_counts
        else:
            input_grad = None

        return (input_grad, weight_grad, bias_grad)


class FlattenLayer:
    def __init__(self, input_size=None):
        self.input_size = input_size
        self.biases = None
        self.weights = None

    @property
    def output_size(self):
        return (np.product(self.input_size),)

    def init_weights(self):
        pass

    def forwards(self, inputs):
        return inputs.flatten()

    def backwards(self, loss_grad, compute_input_grad=True):
        return (loss_grad.reshape(self.input_size), None, None)


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
                layers[i].input_size = layers[i - 1].output_size

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
            if layer.weights is not None:
                sum_weight_grads[layer] = np.zeros(layer.weights.shape)
            if layer.biases is not None:
                sum_bias_grads[layer] = np.zeros(layer.biases.shape)

        total_errors = 0

        for i, (example, label) in enumerate(batch):
            target = onehot(label, max_label + 1)
            output = example

            for layer in self.layers:
                output = layer.forwards(output)

            predicted = np.argmax(output)
            if predicted != label:
                total_errors += 1

            input_grad = loss_op.gradient(target, output)
            for layer in reversed(self.layers):
                compute_input_grad = layer != self.layers[0]
                input_grad, weight_grad, bias_grad = layer.backwards(input_grad,
                                                                     compute_input_grad=compute_input_grad)

                if layer.weights is not None:
                    sum_weight_grads[layer] += weight_grad
                if layer.biases is not None:
                    sum_bias_grads[layer] += bias_grad

        for layer, sum_weight_grad in sum_weight_grads.items():
            if layer.weights is None:
                continue
            mean_grad = sum_weight_grad / len(batch)
            layer.weights = layer.weights - learning_rate * mean_grad

        for layer, sum_bias_grad in sum_bias_grads.items():
            if layer.biases is None:
                continue
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
    train_images, train_labels, test_images, test_labels = load_mnist_dataset(dataset_path, (28, 28))

    model = Model(layers=[
        Conv2DLayer(32, (3, 3), activation=Relu()),
        FlattenLayer(),
        Layer(32, name='relu', activation=Relu()),
        Layer(10, name='softmax', activation=Softmax()),
    ], input_size=(28, 28))

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
