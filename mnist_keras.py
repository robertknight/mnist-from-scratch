"""
Reference MNIST digit classifier using Keras.
"""

import sys

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical

from loader import load_mnist_dataset

if __name__ == '__main__':
    # Load train and test datasets.
    print('reading training data...')
    dataset = sys.argv[1]
    train_images, train_labels, test_images, test_labels = load_mnist_dataset(dataset)
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    # Setup a very simple model for use as a reference when building the
    # "from scratch" implementation.
    model = Sequential()
    model.add(Dense(32, activation='relu', input_shape=(28 * 28,), kernel_initializer='random_uniform'))
    model.add(Dense(10, activation='softmax', kernel_initializer='random_uniform'))

    # Note: The learning rate and decay values that produce optimal results are
    # different between Keras and mnist.py. Finding out why is TBD.
    model.compile(optimizer=SGD(lr=0.02),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    print('training model...')
    model.fit(train_images, train_labels, epochs=10, batch_size=32)

    print('evaluating model...')
    score = model.evaluate(test_images, test_labels, batch_size=128)

    print(f'Score {score}')
