# neuralnet-from-scratch

An implementation of a neural network and a few basic machine learning
algorithms "from scratch" using Python and numpy (ie. not using an existing
deep learning library). The package was created purely as a learning exercise
to help me understand modern machine learning fundamentals.

This repository also includes some simple test programs which use the classic
MNIST dataset.

The training/test datasets are the [classic MNIST dataset](http://yann.lecun.com/exdb/mnist/) and
the more recent, and harder, [fashion MNIST](https://github.com/zalandoresearch/fashion-mnist).

There is also a reference implementation using [Keras](https://keras.io) as a sanity check
and an implementation using [PyTorch](https://pytorch.org)'s low-level APIs.

## Usage

The example programs and neuralnet package require Python 3.6 or later.

```sh
# Install dependencies.
pip install numpy

# Fetch datasets.
./download-data.sh

# Run classifiers.
python mnist_{algorithm}.py data/{classic, fashion}
```

## Variants

File | Description | Test accuracy (classic MNIST) | Test accuracy (fashion MNIST)
--- | --- | --- | ---
`mnist_nn.py -m conv2d` | Simple convolutional neural network. | ~0.99 | ~0.89
`mnist_nn.py -m basic` | 2-layer neural network using ReLU and softmax layers. | ~0.97 | ~0.86
`mnist_logistic_regression.py` | Binary logistic regression using 1-vs-n to handle multi-class classification. | ~0.90 | ~0.82
`mnist_bayes.py` | Multi-class naive bayes. | ~0.83 | ~0.68
