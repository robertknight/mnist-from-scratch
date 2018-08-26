import struct

import numpy as np

MNIST_IMAGE_MAGIC = 2051
MNIST_LABEL_MAGIC = 2049


def read_be32(file):
    data = file.read(4)
    result, *rest = struct.unpack('>l', data)
    return result


def load_mnist_images(path):
    # Data format (big endian):
    # i32 | Magic
    # i32 | Item count
    # i32 | Row count
    # i32 | Column count
    # u8[] | Pixels (organized row-wise)
    with open(path, 'rb') as f:
        magic = read_be32(f)
        if magic != MNIST_IMAGE_MAGIC:
            raise ValueError('Magic number mismatch in image file ({})'.format(magic))
        count = read_be32(f)
        row_count = read_be32(f)
        col_count = read_be32(f)

        images = np.ndarray((count, row_count, col_count),
                            dtype='uint8', buffer=f.read())
        return images


def load_mnist_labels(path):
    # Data format (big endian):
    # i32 | Magic
    # i32 | Item count
    # u8[] | Labels (0-9)
    with open(path, 'rb') as f:
        magic = read_be32(f)
        if magic != MNIST_LABEL_MAGIC:
            raise ValueError('Magic number mismatch in label file ({})'.format(magic))
        count = read_be32(f)
        labels = np.ndarray((count), dtype='uint8', buffer=f.read())
        return labels
