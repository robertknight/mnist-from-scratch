"""
Naive bayes classifier for MNIST data.
"""

import math

import numpy as np

from loader import load_mnist_images, load_mnist_labels


def normal_distribution_pdf(mean, variance, x):
    exp = -((x - mean)**2. / (2. * variance))
    denominator = np.sqrt(2. * math.pi * variance)
    return (math.e ** exp) / denominator


class NaiveBayesClassifier:

    def __init__(self, n_classes, n_features):
        # Prior probabilities for each class.
        self.priors = np.ones(n_classes) / n_classes
        self.n_classes = n_classes
        self.n_features = n_features

        # Means and variances of feature values for each class in the training
        # data.
        self.means = np.zeros((n_classes, n_features))
        self.variances = np.zeros((n_classes, n_features))

    def train(self, examples, labels):
        # Min variance to prevent divide-by-zero in `normal_distribution_pdf`.
        MIN_VARIANCE = 0.1
        MAX_VARIANCE = 1.0

        for label in range(self.n_classes):
            # Get examples from this class.
            class_examples = np.compress(np.equal(labels, label), examples, 0)

            # Compute mean and variance of features in this class.
            self.means[label] = np.mean(class_examples, 0)
            self.variances[label] = np.clip(np.var(class_examples, 0), MIN_VARIANCE, MAX_VARIANCE)

    def predict(self, data):
        label_probs = np.zeros(self.n_classes)

        for label in range(self.n_classes):
            # Compute `p(feature = value | label)`.
            mean = self.means[label]
            variance = self.variances[label]
            feature_probs = normal_distribution_pdf(mean, variance, data)

            # Compute `p(y=k | x) = p(y=k) * p(x=v | y=k)` for each label.
            label_probs[label] = self.priors[label] * np.prod(feature_probs)

        # Return label with highest probability.
        return np.argmax(label_probs)

    def evaluate(self, data, labels):
        errors = 0
        evaluated = 0
        for (x, label) in zip(data, labels):
            y_pred = self.predict(x)
            if y_pred != label:
                errors += 1
            evaluated += 1

        return 1 - (errors / len(data))


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
    model = NaiveBayesClassifier(n_classes=10, n_features=28 * 28)
    model.train(train_images, train_labels)

    print('evaluating...')
    test_accuracy = model.evaluate(test_images, test_labels)
    print(f'test accuracy {test_accuracy}')


if __name__ == '__main__':
    train_and_test()
