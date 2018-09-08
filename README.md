# mnist-from-scratch

Handwritten digit classifiers implemented using various common machine
learning algorithms as a learning exercise. Implementations are "from scatch" using just Python + numpy.

Minimal effort has been put into parameter tuning. I wanted to find out how
well each algorithm performed "out of the box".

The training/test datasets are the [classic MNIST dataset](http://yann.lecun.com/exdb/mnist/) and
the more recent, and harder, [fashion MNIST](https://github.com/zalandoresearch/fashion-mnist).

There is also a reference implementation using [Keras](https://keras.io) as a sanity check.

## Usage

nb. The instructions below assume that "python" is Python >= 3.6.

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
`mnist_nn.py` | 2-layer neural network using ReLU and softmax layers. | ~0.97 | ~0.86
`mnist_logistic_regression.py` | Binary logistic regression using 1-vs-n to handle multi-class classification. | ~0.90 | ~0.82
`mnist_bayes.py` | Multi-class naive bayes. | ~0.83 | ~0.68
