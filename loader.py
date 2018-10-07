import struct

import numpy as np

MNIST_IMAGE_MAGIC = 2051
MNIST_LABEL_MAGIC = 2049


def read_be32(file):
    data = file.read(4)
    result, *rest = struct.unpack(">l", data)
    return result


def load_mnist_images(path):
    # Data format (big endian):
    # i32 | Magic
    # i32 | Item count
    # i32 | Row count
    # i32 | Column count
    # u8[] | Pixels (organized row-wise)
    with open(path, "rb") as f:
        magic = read_be32(f)
        if magic != MNIST_IMAGE_MAGIC:
            raise ValueError("Magic number mismatch in image file ({})".format(magic))
        count = read_be32(f)
        row_count = read_be32(f)
        col_count = read_be32(f)

        images = np.ndarray(
            (count, row_count, col_count), dtype="uint8", buffer=f.read()
        )
        return images


def load_mnist_labels(path):
    # Data format (big endian):
    # i32 | Magic
    # i32 | Item count
    # u8[] | Labels (0-9)
    with open(path, "rb") as f:
        magic = read_be32(f)
        if magic != MNIST_LABEL_MAGIC:
            raise ValueError("Magic number mismatch in label file ({})".format(magic))
        count = read_be32(f)
        labels = np.ndarray((count), dtype="uint8", buffer=f.read())
        return labels


def load_mnist_dataset(path, shape=(28 * 28,)):
    """
    Load an MNIST-format dataset from the given directory.

    The dataset is expected to be in the format described at
    http://yann.lecun.com/exdb/mnist/.
    """
    train_images = load_mnist_images(f"{path}/train-images-idx3-ubyte")
    train_images = train_images.reshape((60000, *shape))
    train_images = train_images.astype("float32") / 255.0
    train_labels = load_mnist_labels(f"{path}/train-labels-idx1-ubyte")

    print("reading test data...")
    test_images = load_mnist_images(f"{path}/t10k-images-idx3-ubyte")
    test_images = test_images.reshape((10000, *shape))
    test_images = test_images.astype("float32") / 255.0
    test_labels = load_mnist_labels(f"{path}/t10k-labels-idx1-ubyte")

    return (train_images, train_labels, test_images, test_labels)
