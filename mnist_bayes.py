"""
Naive bayes classifier for MNIST data.
"""

import math
import sys

import numpy as np

from loader import load_mnist_dataset


def normal_distribution_pdf(mean, variance, x):
    """Normal distribution probability density function (from Wikipedia)."""
    exp = -((x - mean) ** 2.0 / (2.0 * variance))
    denominator = np.sqrt(2.0 * math.pi * variance)
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
            self.variances[label] = np.clip(
                np.var(class_examples, 0), MIN_VARIANCE, MAX_VARIANCE
            )

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
        for x, label in zip(data, labels):
            if self.predict(x) != label:
                errors += 1
        return 1 - (errors / len(data))


def train_and_test(dataset_path):
    print("reading training data...")
    train_images, train_labels, test_images, test_labels = load_mnist_dataset(
        dataset_path
    )

    print("training...")
    model = NaiveBayesClassifier(n_classes=10, n_features=28 * 28)
    model.train(train_images, train_labels)

    print("evaluating...")
    test_accuracy = model.evaluate(test_images, test_labels)
    print(f"test accuracy {test_accuracy}")


if __name__ == "__main__":
    dataset = sys.argv[1]
    train_and_test(dataset)
