from enum import Enum

import numpy as np

from .util import float_type


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

        self.weights = np.random.uniform(
            -0.2, 0.2, (self.unit_count, *self.input_size)
        ).astype(float_type)
        self.biases = np.zeros(self.unit_count, dtype=float_type)

    def forwards(self, inputs, context):
        z = np.dot(self.weights, inputs) + self.biases
        context["inputs"] = inputs
        context["z"] = z
        return self.activation(z)

    def backwards(self, loss_grad, context, compute_input_grad=True):
        """
        Compute the gradients with respect to the loss against a training example.

        :param loss_grad: Gradient of loss wrt. each of this layer's outputs
        :return: 2-tuple of gradient of inputs and weights wrt. loss.
        """
        last_z = context["z"]
        last_inputs = context["inputs"]

        z_grad = self.activation.gradient(last_z, loss_grad)
        bias_grad = z_grad
        weight_grad = np.matmul(
            z_grad.reshape((self.unit_count, 1)),
            last_inputs.reshape((1, *self.input_size)),
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

    windows = np.zeros(
        (filter_h, filter_w, input_channels, *window_size), dtype=float_type
    )
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
        ).astype(float_type)

    def forwards(self, inputs, context):
        assert inputs.shape == self.input_size

        if self.padding == Padding.SAME:
            vpad = self.filter_shape[0] // 2
            hpad = self.filter_shape[1] // 2
            inputs = np.pad(
                inputs, pad_width=((0, 0), (vpad, vpad), (hpad, hpad)), mode="constant"
            )

        conv2d_outputs = conv2d(inputs, self.weights)
        channel_outputs = self.activation(conv2d_outputs)

        context["inputs"] = inputs
        context["conv2d_outputs"] = conv2d_outputs

        return channel_outputs

    def backwards(self, loss_grad, context, compute_input_grad=True):
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

        last_conv2d_outputs = context["conv2d_outputs"]
        last_inputs = context["inputs"]

        # Compute gradients of activation input wrt. loss.
        activation_grads = self.activation.gradient(last_conv2d_outputs, loss_grad)

        # Compute gradient of weights wrt. loss.
        weight_grad = np.zeros(self.weights.shape, float_type)

        # Compute weight gradient for each element of filter.
        # The filter is typically much smaller than the input so we get
        # more efficient vectorization than looping over windows in the input.
        input_windows = filter_windows(last_inputs, self.weights)
        weight_grad = np.einsum("yxDij,Cij->CDyx", input_windows, activation_grads)
        assert weight_grad.shape == self.weights.shape

        if compute_input_grad:
            # Gradients of inputs wrt. loss.
            input_grad = np.zeros(padded_input_size, float_type)
            # Count of weights that were multiplied by each input position.
            # Input positions in the center of the image are multiplied by all
            # (y, x) elements of the kernel. Positions near the edge are
            # multiplied by fewer elements when using "valid" padding.
            weight_counts = np.zeros((input_channels, input_h, input_w), float_type)
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
                        np.ones(weight_count_window.shape, dtype=float_type)
                        * self.channels
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

    def forwards(self, inputs, context):
        return inputs.flatten()

    def backwards(self, loss_grad, context, compute_input_grad=True):
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

    def forwards(self, inputs, context):
        pool_width, pool_height = self.window_size
        output = np.zeros(self.output_size, float_type)
        _, output_h, output_w = self.output_size

        for h in range(pool_height):
            for w in range(pool_width):
                pool_view = inputs[:, h::pool_height, w::pool_width]
                pool_view = self._clip_to_output_size(pool_view)
                output = np.maximum(output, pool_view)

        context["inputs"] = inputs
        context["output"] = output

        return output

    def backwards(self, loss_grad, context, compute_input_grad=True):
        pool_width, pool_height = self.window_size
        _, output_h, output_w = self.output_size
        last_inputs = context["inputs"]
        last_output = context["output"]

        input_grad = np.zeros(self.input_size, float_type)

        for h in range(pool_height):
            for w in range(pool_width):
                input_view = last_inputs[:, h::pool_height, w::pool_width]
                input_view = self._clip_to_output_size(input_view)
                input_grad_view = input_grad[:, h::pool_height, w::pool_width]
                input_grad_view = self._clip_to_output_size(input_grad_view)
                mask = np.equal(input_view, last_output)
                np.copyto(input_grad_view, loss_grad, where=mask)

        return (input_grad, None, None)

    def _clip_to_output_size(self, array):
        _, output_h, output_w = self.output_size
        return array[:, :output_h, :output_w]
