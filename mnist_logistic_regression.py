"""
Logistic regression classifier for MNIST.
"""

import numpy as np

from loader import load_mnist_images, load_mnist_labels


def sigmoid(x):
    return 1. / (1 + np.exp(-x))


def sigmoid_grad(x):
    exp_mx = np.exp(-x)
    return exp_mx / (exp_mx + 1)**2


def log_loss_grad(x, y):
    """Gradient of `-log(x)` (if y is 1) or `-log(1 - x)` (if y is 0"""
    if y:
        return -1. / x
    else:
        return 1. / (1 - x)


class LogisticRegressionClassifier:

    def __init__(self, n_features):
        self.n_features = n_features
        self.weights = None
        self.bias = 0.

    def fit(self, examples, labels):
        learning_rate = 0.005
        self.weights = np.random.uniform(-0.1, 0.1, self.n_features)
        self.bias = 0.

        for example, label in zip(examples, labels):
            z = np.dot(self.weights, example) + self.bias
            activation = sigmoid(z)
            loss_grad = log_loss_grad(activation, label)
            z_grad = sigmoid_grad(z) * loss_grad
            weight_grad = example * z_grad
            bias_grad = z_grad

            self.weights = self.weights - learning_rate * weight_grad
            self.bias = self.bias - learning_rate * bias_grad

    def predict(self, data):
        z = np.dot(self.weights, data) + self.bias
        return sigmoid(z)


def train_and_test():
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

    print('training...')
    models = []
    for label in range(10):
        model = LogisticRegressionClassifier(n_features=28 * 28)
        class_labels = np.where(train_labels == label, 1., 0.)
        model.fit(train_images, class_labels)
        models.append(model)

    print('evaluating...')
    errors = 0
    for example, label in zip(test_images, test_labels):
        scores = [model.predict(example) for model in models]
        predicted_label = np.argmax(scores)

        if predicted_label != label:
            errors += 1
    accuracy = 1. - (errors / len(test_images))

    print(f'test accuracy {accuracy}')


if __name__ == '__main__':
    train_and_test()
