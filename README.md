# mnist-from-scratch

Handwritten digit classifiers implemented using various common machine
learning algorithms as a learning exercise. Implementations are "from scatch" using just Python + numpy.

Minimal effort has been put into parameter tuning. I wanted to find out how
what the "out of the box" performance was like for each algorithm.

The training/test data is the standard MNIST dataset from http://yann.lecun.com/exdb/mnist/

There is also a reference implementation using [Keras](https://keras.io) as a sanity check.

## Usage

nb. The instructions below assume that "python" is Python >= 3.6.

```
pip install numpy
python mnist_{algorithm}.py
```

## Variants

File | Description | Test accuracy
--- | --- | ---
`mnist_nn.py` | 2-layer neural network using ReLU and softmax layers. | ~0.97
`mnist_logistic_regression.py` | Binary logistic regression using 1-vs-n to handle multi-class classification. | ~0.89
`mnist_bayes.py` | Multi-class naive bayes. | ~0.83
