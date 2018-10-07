"""
MNIST handwritten digit classifier neural net.
"""

import argparse

import numpy as np

from loader import load_mnist_dataset
from neuralnet.model import Model
from neuralnet.layers import Conv2DLayer, MaxPoolingLayer, Layer, FlattenLayer
from neuralnet.ops import Relu, Softmax, CategoricalCrossentropy


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
