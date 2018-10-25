"""
Handwritten digit classifier using low-level PyTorch APIs.

This implementation was created to familiarize myself with low-level
PyTorch APIs and in particular the autograd package. It is inefficient
and lengthy compared to using PyTorch's built-in neural net modules.
"""

import torch
from torch import Tensor

from loader import load_mnist_dataset


def uniform(shape, min_, max_, requires_grad=False):
    result = torch.rand(shape) * (max_ - min_) + min_
    result.requires_grad_(requires_grad)
    return result


def relu(x):
    return torch.max(torch.tensor(0.0), x)


def softmax(x):
    shifted_x = x - torch.max(x)
    exp_s = torch.exp(shifted_x)
    return exp_s / torch.sum(exp_s)


def onehot(x, length):
    y = torch.zeros(length)
    y[x] = 1.0
    return y


def crossentropy(targets, predictions):
    targets = torch.clamp(targets, 1e-7, 1.0)
    predictions = torch.clamp(predictions, 1e-7, 1.0)
    return -torch.sum(targets * torch.log(predictions))


class DenseLayer:
    def __init__(self, units, input_shape, activation, learning_rate=0.01):
        self.weights = uniform((units, *input_shape), -0.2, 0.2, requires_grad=True)
        self.biases = torch.zeros(units, requires_grad=True)
        self.activation = activation
        self.learning_rate = learning_rate

    def forwards(self, input_):
        return self.activation(self.weights @ input_ + self.biases)

    def update_weights(self):
        self.weights.data.add_(self.learning_rate * -self.weights.grad)
        self.biases.data.add_(self.learning_rate * -self.biases.grad)
        self.weights.grad.zero_()
        self.biases.grad.zero_()


def train(images, labels) -> (Tensor, Tensor):
    image_count, features = images.shape
    classes = 10
    image_count = images.shape[0]

    epochs = 3
    batch_size = 32
    learning_rate = 0.01

    layers = [
        DenseLayer(32, (28 * 28,), activation=relu, learning_rate=learning_rate),
        DenseLayer(10, (32,), activation=softmax, learning_rate=learning_rate),
    ]

    for epoch in range(epochs):
        print(f"epoch {epoch + 1}")
        for i in range(0, image_count, batch_size):
            for minibatch_index in range(0, batch_size):
                image = images[i + minibatch_index]
                label = labels[i + minibatch_index]

                output = image
                for layer in layers:
                    output = layer.forwards(output)
                loss = crossentropy(onehot(label.item(), classes), output)
                loss.backward()

            for layer in layers:
                layer.update_weights()

    return layers


@torch.no_grad()
def test(layers, images, labels):
    correct_predictions = 0.0
    total_predictions = 0.0

    for image, label in zip(images, labels):
        output = image
        for layer in layers:
            output = layer.forwards(output)
        pred = torch.argmax(output)
        total_predictions += 1
        if pred.item() == label.item():
            correct_predictions += 1

    return correct_predictions / total_predictions


train_images, train_labels, test_images, test_labels = load_mnist_dataset(
    "data/classic"
)
train_images, train_labels, test_images, test_labels = (
    torch.from_numpy(train_images),
    torch.from_numpy(train_labels),
    torch.from_numpy(test_images),
    torch.from_numpy(test_labels),
)

print(f"training...")
layers = train(train_images, train_labels)
print(f"testing...")
accuracy = test(layers, test_images, test_labels)
print(f"accuracy: {accuracy}")
