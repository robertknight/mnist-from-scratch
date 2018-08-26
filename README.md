# mnist-from-scratch

A simple neural network-based classifier for handwritten digits, implemented
"from scatch" (ie. using only Python + numpy, not any existing deep learning
framework) as a learning exercise.

The training/test data is the standard MNIST dataset from http://yann.lecun.com/exdb/mnist/

## Usage

nb. The instructions below assume that "python" is Python >= 3.6.

```
pip install numpy
python mnist.py
```

The accuracy results should be similar to the reference implementation using
[Keras](https://keras.io):

```
pip install keras
python mnist_keras.py
```
