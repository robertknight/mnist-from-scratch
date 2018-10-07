import concurrent.futures as futures
from multiprocessing import cpu_count
import time

import numpy as np

from .util import onehot, shuffle_examples


class ProgressReporter:
    def report_training_progress(
        self,
        epoch,
        total_examples,
        examples_processed,
        epoch_start_time,
        epoch_total_errors,
        is_last_batch,
    ):
        now = time.time()
        time_per_example = (now - epoch_start_time) / examples_processed
        time_per_example_us = time_per_example * 1_000_000
        accuracy = 1 - (epoch_total_errors / examples_processed)
        print("\r", end="")
        print(
            f"epoch {epoch} ({examples_processed} / {total_examples})  "
            f"{time_per_example_us:.2f}us/example",
            f"  accuracy {accuracy:.3f}",
            end="",
        )
        if is_last_batch:
            print("\n", end="")


class Model:
    """
    Simple neural network model consisting of a stack of layers.
    """

    def __init__(self, layers, input_size, progress_reporter=ProgressReporter()):
        self.layers = layers
        self.progress_reporter = progress_reporter

        for i in range(0, len(layers)):
            if i == 0:
                layers[0].input_size = input_size
            else:
                layers[i].input_size = layers[i - 1].output_size

        self.executor = futures.ThreadPoolExecutor(max_workers=cpu_count())

    def fit(
        self,
        data,
        labels,
        batch_size,
        epochs,
        learning_rate,
        loss_op,
        learning_rate_decay=0.0,
    ):

        """Learn parameters given input training `data` and target `labels`."""

        data, labels = shuffle_examples(data, labels)
        reporter = self.progress_reporter

        # Reset model.
        for layer in self.layers:
            layer.init_weights()

        # Divide training set into mini-batches.
        max_label = np.max(labels)
        batches = []
        for offset in range(0, len(data), batch_size):
            batches.append(
                (
                    data[offset : offset + batch_size],
                    labels[offset : offset + batch_size],
                )
            )

        # Train model.
        for epoch in range(0, epochs):
            epoch_errors = 0
            epoch_start_time = time.time()
            for batch_index, (batch_data, batch_labels) in enumerate(batches):
                shuffled_batch = list(zip(batch_data, batch_labels))
                np.random.shuffle(shuffled_batch)
                batch_errors = self._fit_batch(
                    shuffled_batch, max_label, loss_op, learning_rate
                )
                epoch_errors += batch_errors

                reporter.report_training_progress(
                    epoch=epoch,
                    total_examples=len(data),
                    examples_processed=(batch_index + 1) * batch_size,
                    epoch_start_time=epoch_start_time,
                    epoch_total_errors=epoch_errors,
                    is_last_batch=batch_index == len(batches) - 1,
                )
            learning_rate *= 1 - learning_rate_decay

    def _fit_batch(self, batch, max_label, loss_op, learning_rate):
        # Compute sum of cost gradients across all examples in batch.
        sum_weight_grads = {}
        sum_bias_grads = {}

        for layer in self.layers:
            if layer.weights is not None:
                sum_weight_grads[layer] = np.zeros(layer.weights.shape, np.float32)
            if layer.biases is not None:
                sum_bias_grads[layer] = np.zeros(layer.biases.shape, np.float32)

        total_errors = 0

        def fit_example(example, label):
            target = onehot(label, max_label + 1)
            output = example
            context = {}

            for layer in self.layers:
                context[layer] = {}
                output = layer.forwards(output, context[layer])

            # Make sure that all of the calculations produced single-precision
            # results.
            assert output.dtype == np.float32

            predicted = np.argmax(output)
            if predicted != label:
                nonlocal total_errors
                total_errors += 1

            input_grad = loss_op.gradient(target, output)
            for layer in reversed(self.layers):
                compute_input_grad = layer != self.layers[0]
                input_grad, weight_grad, bias_grad = layer.backwards(
                    input_grad, context[layer], compute_input_grad=compute_input_grad
                )

                if layer.weights is not None:
                    sum_weight_grads[layer] += weight_grad
                if layer.biases is not None:
                    sum_bias_grads[layer] += bias_grad

        tasks = [
            self.executor.submit(fit_example, example, label)
            for example, label in batch
        ]
        done, *rest = futures.wait(tasks)
        # Re-raise any exceptions
        [task.result() for task in done]

        for layer, sum_weight_grad in sum_weight_grads.items():
            if layer.weights is None:
                continue
            mean_grad = sum_weight_grad / len(batch)
            layer.weights = layer.weights - learning_rate * mean_grad

        for layer, sum_bias_grad in sum_bias_grads.items():
            if layer.biases is None:
                continue
            mean_grad = sum_bias_grad / len(batch)
            layer.biases = layer.biases - learning_rate * mean_grad

        return total_errors

    def predict(self, features):
        """
        Predict the class of an example given a feature vector.

        Returns the predicted class label.
        """
        output = features
        for layer in self.layers:
            output = layer.forwards(output, context={})
        return np.argmax(output)

    def evaluate(self, data, target_labels):
        """
        Evaluate model on test data.

        Return the accuracy (proportion of correctly classified examples).
        """
        errors = 0
        for i, features in enumerate(data):
            predicted_class = self.predict(features)
            actual_class = target_labels[i]
            if predicted_class != actual_class:
                errors += 1
        return 1 - (errors / len(data))
