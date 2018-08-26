"""
Reference MNIST digit classifier using Keras.
"""

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical

from loader import load_mnist_images, load_mnist_labels

if __name__ == '__main__':
    # Load train and test datasets.
    # These are in the original form provided by http://yann.lecun.com/exdb/mnist/
    print('reading training data...')
    train_images = load_mnist_images('data/train-images.idx3-ubyte')
    train_images = train_images.reshape((60000, 28 * 28))
    train_images = train_images.astype('float') / 255.0
    train_labels = load_mnist_labels('data/train-labels.idx1-ubyte')
    train_labels = to_categorical(train_labels)

    print('reading test data...')
    test_images = load_mnist_images('data/t10k-images.idx3-ubyte')
    test_images = test_images.reshape((10000, 28 * 28))
    test_images = test_images.astype('float') / 255.0
    test_labels = load_mnist_labels('data/t10k-labels.idx1-ubyte')
    test_labels = to_categorical(test_labels)

    # Setup a very simple model for use as a reference when building the
    # "from scratch" implementation.
    print('building model...')
    model = Sequential()
    model.add(Dense(10, activation='softmax', input_shape=(28 * 28,), use_bias=False, kernel_initializer='random_uniform'))
    model.compile(optimizer=SGD(lr=0.01),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    print('training model...')
    model.fit(train_images, train_labels, epochs=10, batch_size=32)

    print('evaluating model...')
    score = model.evaluate(test_images, test_labels, batch_size=128)

    print(f'Score {score}')
