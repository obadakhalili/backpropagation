import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from time import perf_counter
from functools import reduce


def build_fully_connected_network(
    layers_sizes,
    activation,
    activation_derivative,
    training_data,
    mini_batch_size,
    epochs_count=100,
    on_epoch_end=lambda epoch_number, learning_time, model: None,
    learning_rate=0.1,
):
    weights = [
        np.random.randn(next_layer_size, layer_size)
        for layer_size, next_layer_size in zip(layers_sizes[:-1], layers_sizes[1:])
    ]
    biases = [np.random.randn(layer_size, 1) for layer_size in layers_sizes[1:]]

    def feedforward(activations):
        nonlocal weights, biases

        for weights, biases in zip(weights, biases):
            activations = activation(np.dot(weights, activations) + biases)

        return activations

    def model(inputs_activations):
        return feedforward(inputs_activations.reshape(inputs_activations.shape[0], -1))

    def backprop(x, y):
        pass

    def SGD():
        nonlocal weights, biases

        for epoch_number in range(epochs_count):
            np.random.shuffle(training_data)
            start_time = perf_counter()

            for mini_batch in [
                training_data[mini_batch_idx : mini_batch_idx + mini_batch_size]
                for mini_batch_idx in range(0, len(training_data), mini_batch_size)
            ]:
                pass

            on_epoch_end(epoch_number, perf_counter() - start_time, model)

    # learn
    SGD()

    return model


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(z):
    sigmoid_given_z = sigmoid(z)
    return sigmoid_given_z * (1 - sigmoid_given_z)


def recognize_digit(model, inputs_activations):
    return np.argmax(model(inputs_activations))


def main():
    data = datasets.load_digits()
    X_train, X_test, y_train, y_test = train_test_split(
        data.images.reshape((data.images.shape[0], -1)),
        data.target,
        test_size=0.3,
        shuffle=False,
    )
    pixels_count = data.images.shape[1] * data.images.shape[2]

    def evaluate_model(model):
        return reduce(
            lambda accuracy, sample: (
                accuracy
                + (recognize_digit(model, sample[0]) == sample[1]) / len(y_test)
            ),
            zip(X_test, y_test),
            0,
        )

    model = build_fully_connected_network(
        layers_sizes=[pixels_count, 3, 10],
        activation=sigmoid,
        activation_derivative=sigmoid_derivative,
        training_data=list(zip(X_train, y_train)),
        mini_batch_size=50,
        on_epoch_end=lambda epoch_number, learning_time, model: print(
            f"Epoch {epoch_number} finished in {learning_time} with accuracy: {evaluate_model(model)}"
        ),
    )

    print(f"Final accuracy: {evaluate_model(model)}")


if __name__ == "__main__":
    main()
