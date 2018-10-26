"""
Handwritten digit classifier using low-level Tensorflow APIs.

This implementation was created to familiarize myself with low-level
Tensorflow APIs and the graph programming model. Hence it doesn't make use of
Tensorflow's high level APIs, which would make for a much shorter solution!
"""

import numpy as np
import tensorflow as tf

from loader import load_mnist_dataset


def relu(x):
    return tf.maximum(0.0, x)


def softmax(x):
    shifted_x = x - tf.reduce_max(x, axis=-1, keepdims=True)
    exp_s = tf.exp(shifted_x)
    return exp_s / tf.reduce_sum(exp_s, axis=-1, keepdims=True)


def crossentropy(targets, predictions):
    targets = tf.clip_by_value(targets, 1e-7, 1.0)
    predictions = tf.clip_by_value(predictions, 1e-7, 1.0)
    return -tf.reduce_sum(targets * tf.log(predictions), axis=-1)


class DenseLayer:
    def __init__(self, units, input_shape, activation):
        self.weights = tf.Variable(
            tf.random_uniform((units, *input_shape), -0.2, 0.2), name="weights"
        )
        self.biases = tf.Variable(tf.zeros(units), name="biases")
        self.activation = activation
        self.units = units

    def forwards(self, input_):
        z = input_ @ tf.transpose(self.weights)
        z = z + tf.reshape(self.biases, (1, self.units))
        return self.activation(z)

    def update_weights(self, loss, learning_rate):
        weight_grad, = tf.gradients(xs=self.weights, ys=loss)
        bias_grad, = tf.gradients(xs=self.biases, ys=loss)
        update_weights = tf.assign_sub(self.weights, learning_rate * weight_grad)
        update_biases = tf.assign_sub(self.biases, learning_rate * bias_grad)
        return tf.group(update_weights, update_biases)


def train(session, images, labels, batch_size=32):
    """
    Create a TensorFlow graph for a simple neural net and train parameters.

    Returns an (input, output) tf.Tensor tuple.
    """
    classes = 10
    input_shape = (28 * 28,)
    learning_rate = 0.01
    epochs = 5
    image_count = images.shape[0]

    # Placeholders for input and output.
    image_input = tf.placeholder(
        tf.float32, shape=(batch_size, *input_shape), name="image_input"
    )
    label_input = tf.placeholder(
        tf.float32, shape=(batch_size, classes), name="label_input"
    )

    # Construct a simple TensorFlow model.
    layers = [
        DenseLayer(20, input_shape, activation=relu),
        DenseLayer(classes, (20,), activation=softmax),
    ]
    output = image_input
    for layer in layers:
        output = layer.forwards(output)
    loss = crossentropy(label_input, output)

    train = tf.group(
        *[layer.update_weights(loss, learning_rate=learning_rate) for layer in layers]
    )

    # Finalize model.
    session.run(tf.global_variables_initializer())

    # Train model for several epochs.
    onehot_labels = np.zeros((image_count, classes), dtype=np.float32)
    for i in range(image_count):
        onehot_labels[i][labels[i]] = 1.0

    for epoch in range(epochs):
        for i in range(0, image_count, batch_size):
            batch_images = images[i : i + batch_size]
            batch_labels = onehot_labels[i : i + batch_size]

            session.run(
                train, feed_dict={image_input: batch_images, label_input: batch_labels}
            )

    return (image_input, output)


def test(session, input_, output, images, labels, batch_size):
    total_predictions = 0.0
    correct_predictions = 0.0
    image_count = images.shape[0]

    for i in range(0, image_count, batch_size):
        image_batch = images[i : i + batch_size]

        # Pad the number of examples in the batch to the batch size.
        padded_image_batch = image_batch
        extra_rows = batch_size - image_batch.shape[0]
        if extra_rows > 0:
            padded_image_batch = np.pad(
                image_batch, ((0, extra_rows), (0, 0)), mode="edge"
            )

        prediction_batch = session.run(output, feed_dict={input_: padded_image_batch})

        for k in range(0, image_batch.shape[0]):
            predicted_class = np.argmax(prediction_batch[k])
            total_predictions += 1
            if predicted_class == labels[i + k]:
                correct_predictions += 1

    return correct_predictions / total_predictions


train_images, train_labels, test_images, test_labels = load_mnist_dataset(
    "data/classic"
)

with tf.Session() as sess:
    batch_size = 32
    input_, output = train(sess, train_images, train_labels, batch_size=batch_size)
    accuracy = test(
        sess, input_, output, test_images, test_labels, batch_size=batch_size
    )
    print(f"accuracy {accuracy}")
