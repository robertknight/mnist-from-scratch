"""
MNIST handwritten digit classifier neural net.
"""

import argparse
from enum import Enum
import random
import time

import numpy as np

from loader import load_mnist_dataset

"""
Just-above-zero value used to avoid zero in places where that would cause undefined results.
"""
EPSILON = 1e-7


def onehot(x, length):
    xs = np.zeros(length)
    xs[x] = 1.0
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
        return np.where(x >= 0.0, 1.0, 0.0) * grads


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


class ProgressReporter:
    def report_training_progress(
        self,
        epoch,
        total_examples,
        examples_processed,
        epoch_start_time,
        epoch_total_errors,
        is_last_batch,
    ):
        now = time.time()
        time_per_example = (now - epoch_start_time) / examples_processed
        time_per_example_us = time_per_example * 1_000_000
        accuracy = 1 - (epoch_total_errors / examples_processed)
        print("\r", end="")
        print(
            f"epoch {epoch} ({examples_processed} / {total_examples})  "
            f"{time_per_example_us:.2f}us/example",
            f"  accuracy {accuracy:.3f}",
            end="",
        )
        if is_last_batch:
            print("\n", end="")


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
        z_grad = self.activation.gradient(self.last_z, loss_grad)
        bias_grad = z_grad
        weight_grad = np.matmul(
            z_grad.reshape((self.unit_count, 1)),
            self.last_inputs.reshape((1, *self.input_size)),
        )

        if compute_input_grad:
            input_grad = np.matmul(np.transpose(self.weights), z_grad)
        else:
            input_grad = None

        return (input_grad, weight_grad, bias_grad)


def filter_windows(input_, filter_):
    """
    Split input into slices to be multiplied with elements of a convolution filter.

    Returns a 5D tensor (filter_rows, filter_columns, input_channels, input_row,
    input_col) where the first two dimensions are positions in a convolution
    filter and the remaining dimensions are a slice of `input_` which will be
    multiplied by that element of the filter during convolution.

    A convolution is normally expressed as sliding a (typically small) filter
    over a (typically larger) input image. Another way to calculate the result
    is to create (filter_height * filter_width) windows over the input where
    each window is the subsection of the input that is multiplied by that filter
    element. These windows are then multiplied by the corresponding filter
    element and summed to produce the output.

    Doing this enables convolution, weight and input gradient calculations to
    be done with a single `np.einsum` call and fewer Python loops, which makes
    training much faster.
    """

    _, _, filter_h, filter_w = filter_.shape
    input_channels, input_h, input_w = input_.shape

    window_size = (input_h - filter_h + 1, input_w - filter_w + 1)

    windows = np.zeros((filter_h, filter_w, input_channels, *window_size))
    for y in range(filter_h):
        for x in range(filter_w):
            windows[y][x] = input_[
                :, y : input_h - filter_h + y + 1, x : input_w - filter_w + x + 1
            ]
    return windows


def conv2d(input_, filter_):
    """
    Return a 2D convolution of an image with a filter.

    :param input_: 3D ndarray of [channel, row, column]
    :param filter_: 4D ndarray of [output channel, input channel, row, column]
    """

    input_windows = filter_windows(input_, filter_)
    return np.einsum("CDyx,yxDij->Cij", filter_, input_windows)


class Padding(Enum):
    VALID = 0
    SAME = 1


class Conv2DLayer:
    """
    A 2D convolution layer.
    """

    def __init__(
        self,
        channels,
        filter_shape,
        activation,
        padding=Padding.VALID,
        input_size=None,
        name=None,
    ):
        """
        :param channels: Number of output channels
        :param filter_shape: Convolution kernel size
        :param activation: Activation applied to convolution outputs
        :param input_size: 3-tuple of (channels, rows, columns)
        :param name: Layer name (for debugging etc.)
        """
        self.channels = channels
        self.filter_shape = filter_shape
        self.activation = activation
        self.input_size = input_size
        self.biases = None
        self.name = name
        self.padding = padding

    @property
    def output_size(self):
        _, input_h, input_w = self.input_size
        if self.padding == Padding.VALID:
            output_shape = np.subtract((input_h, input_w), self.filter_shape) + 1
        else:
            output_shape = (input_h, input_w)
        return (self.channels, *output_shape)

    def init_weights(self):
        assert self.input_size is not None

        input_channels, *rest = self.input_size

        self.biases = None
        self.weights = np.random.uniform(
            -0.2, 0.2, (self.channels, input_channels, *self.filter_shape)
        )

    def forwards(self, inputs):
        assert inputs.shape == self.input_size

        if self.padding == Padding.SAME:
            vpad = self.filter_shape[0] // 2
            hpad = self.filter_shape[1] // 2
            inputs = np.pad(
                inputs, pad_width=((0, 0), (vpad, vpad), (hpad, hpad)), mode="constant"
            )

        conv2d_outputs = conv2d(inputs, self.weights)
        channel_outputs = self.activation(conv2d_outputs)

        self.last_conv2d_outputs = conv2d_outputs
        self.last_inputs = inputs

        return channel_outputs

    def backwards(self, loss_grad, compute_input_grad=True):
        assert loss_grad.shape == self.output_size

        filter_w, filter_h = self.filter_shape
        if self.padding == Padding.SAME:
            vpad = self.filter_shape[0] // 2
            hpad = self.filter_shape[1] // 2
            padded_input_size = np.add(self.input_size, (0, vpad * 2, hpad * 2))
        else:
            vpad = None
            hpad = None
            padded_input_size = self.input_size
        input_channels, input_w, input_h = padded_input_size

        bias_grad = None  # Not implemented yet.

        # Compute gradients of activation input wrt. loss.
        activation_grads = self.activation.gradient(self.last_conv2d_outputs, loss_grad)

        # Compute gradient of weights wrt. loss.
        weight_grad = np.zeros(self.weights.shape)

        # Compute weight gradient for each element of filter.
        # The filter is typically much smaller than the input so we get
        # more efficient vectorization than looping over windows in the input.
        input_windows = filter_windows(self.last_inputs, self.weights)
        weight_grad = np.einsum("yxDij,Cij->CDyx", input_windows, activation_grads)
        assert weight_grad.shape == self.weights.shape

        if compute_input_grad:
            # Gradients of inputs wrt. loss.
            input_grad = np.zeros(padded_input_size)
            # Count of weights that were multiplied by each input position.
            # Input positions in the center of the image are multiplied by all
            # (y, x) elements of the kernel. Positions near the edge are
            # multiplied by fewer elements when using "valid" padding.
            weight_counts = np.zeros((input_channels, input_h, input_w))
            input_grads = np.einsum("CDyx,Cij->yxDij", self.weights, activation_grads)
            for y in range(filter_h):
                for x in range(filter_w):
                    input_grad_window = input_grad[
                        :,
                        y : input_h - filter_h + y + 1,
                        x : input_w - filter_w + x + 1,
                    ]
                    np.copyto(input_grad_window, input_grads[y, x])

                    weight_count_window = weight_counts[
                        :,
                        y : input_h - filter_h + y + 1,
                        x : input_w - filter_w + x + 1,
                    ]
                    weight_count_window += (
                        np.ones(weight_count_window.shape) * self.channels
                    )
            input_grad /= weight_counts
        else:
            input_grad = None

        if self.padding == Padding.SAME:
            input_grad = input_grad[:, vpad:-vpad, hpad:-hpad]

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


class MaxPoolingLayer:
    def __init__(self, window_size=(2, 2), input_size=None):
        self.input_size = input_size
        self.biases = None
        self.weights = None
        self.window_size = window_size

    @property
    def output_size(self):
        channels, width, height = self.input_size
        pool_width, pool_height = self.window_size

        return (channels, width // pool_width, height // pool_height)

    def init_weights(self):
        pass

    def forwards(self, inputs):
        pool_width, pool_height = self.window_size
        output = np.zeros(self.output_size)
        _, output_h, output_w = self.output_size

        for h in range(pool_height):
            for w in range(pool_width):
                pool_view = inputs[:, h::pool_height, w::pool_width]
                pool_view = self._clip_to_output_size(pool_view)
                output = np.maximum(output, pool_view)

        self.last_inputs = inputs
        self.last_output = output

        return output

    def backwards(self, loss_grad, compute_input_grad=True):
        pool_width, pool_height = self.window_size
        _, output_h, output_w = self.output_size

        input_grad = np.zeros(self.input_size)

        for h in range(pool_height):
            for w in range(pool_width):
                input_view = self.last_inputs[:, h::pool_height, w::pool_width]
                input_view = self._clip_to_output_size(input_view)
                input_grad_view = input_grad[:, h::pool_height, w::pool_width]
                input_grad_view = self._clip_to_output_size(input_grad_view)
                mask = np.equal(input_view, self.last_output)
                np.copyto(input_grad_view, loss_grad, where=mask)

        return (input_grad, None, None)

    def _clip_to_output_size(self, array):
        _, output_h, output_w = self.output_size
        return array[:, :output_h, :output_w]


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

    def fit(
        self,
        data,
        labels,
        batch_size,
        epochs,
        learning_rate,
        loss_op,
        learning_rate_decay=0.0,
    ):

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
            batches.append(
                (
                    data[offset : offset + batch_size],
                    labels[offset : offset + batch_size],
                )
            )

        # Train model.
        for epoch in range(0, epochs):
            epoch_errors = 0
            epoch_start_time = time.time()
            for batch_index, (batch_data, batch_labels) in enumerate(batches):
                shuffled_batch = list(zip(batch_data, batch_labels))
                np.random.shuffle(shuffled_batch)
                batch_errors = self._fit_batch(
                    shuffled_batch, max_label, loss_op, learning_rate
                )
                epoch_errors += batch_errors

                reporter.report_training_progress(
                    epoch=epoch,
                    total_examples=len(data),
                    examples_processed=(batch_index + 1) * batch_size,
                    epoch_start_time=epoch_start_time,
                    epoch_total_errors=epoch_errors,
                    is_last_batch=batch_index == len(batches) - 1,
                )
            learning_rate *= 1 - learning_rate_decay

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

        for example, label in batch:
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
                input_grad, weight_grad, bias_grad = layer.backwards(
                    input_grad, compute_input_grad=compute_input_grad
                )

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


def train_and_test(dataset_path, model="basic"):
    # Debugging.
    np.seterr(divide="raise")

    if model == "conv2d":
        # Load train and test datasets.
        print("reading data...")
        train_images, train_labels, test_images, test_labels = load_mnist_dataset(
            dataset_path, (1, 28, 28)
        )

        model = Model(
            layers=[
                Conv2DLayer(32, (3, 3), activation=Relu()),
                MaxPoolingLayer((2, 2)),
                Conv2DLayer(32, (3, 3), activation=Relu()),
                MaxPoolingLayer((2, 2)),
                FlattenLayer(),
                Layer(64, name="relu", activation=Relu()),
                Layer(10, name="softmax", activation=Softmax()),
            ],
            input_size=(1, 28, 28),
        )

        print("training model...")
        model.fit(
            train_images,
            train_labels,
            batch_size=32,
            epochs=10,
            learning_rate=0.15,
            learning_rate_decay=0.1,
            loss_op=CategoricalCrossentropy(),
        )
    elif model == "basic":
        input_size = (28 * 28,)
        train_images, train_labels, test_images, test_labels = load_mnist_dataset(
            dataset_path, input_size
        )

        model = Model(
            layers=[
                Layer(32, name="relu", activation=Relu()),
                Layer(10, name="softmax", activation=Softmax()),
            ],
            input_size=input_size,
        )

        print("training model...")
        model.fit(
            train_images,
            train_labels,
            batch_size=32,
            epochs=10,
            learning_rate=0.18,
            learning_rate_decay=0.1,
            loss_op=CategoricalCrossentropy(),
        )
    else:
        raise Exception(f"Unknown model {model}")

    print("evaluating model...")
    accuracy = model.evaluate(test_images, test_labels)
    print("accuracy {:3f}".format(accuracy))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m, --model",
        dest="model",
        choices=("basic", "conv2d"),
        default="basic",
        help="Network model to use",
    )
    parser.add_argument("dataset_path", help="Path to MNIST-formatted dataset")
    args = parser.parse_args()

    train_and_test(dataset_path=args.dataset_path, model=args.model)
