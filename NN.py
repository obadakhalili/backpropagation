import numpy as np
from time import perf_counter


def FCNN(
    training_data,
    test_data,
    layers_sizes,
    layers_activation_funcs,
    layers_activations_derivatives,
    batch_size,
    seed=None,
    learning_rate=0.1,
    epochs=100,
    on_epoch_end=lambda epoch_number, learning_time, model: None,
):
    if seed:
        np.random.seed(seed)

    weights = [
        np.random.randn(layer_b_size, layer_a_size)
        for layer_a_size, layer_b_size in zip(
            [training_data.shape[1], *layers_sizes[:-1]], layers_sizes
        )
    ]
    biases = [np.random.randn(layer_size, 1) for layer_size in layers_sizes]

    def feedforward(inputs):
        activations = inputs.reshape(-1, inputs.shape[1], 1)

        for layer_index, params in enumerate(zip(weights, biases)):
            layer_weights, layer_biases = params

            return print(activations.T @ layer_weights)
            # https://linux-blog.anracom.com/2019/08/31/numpy-matrix-multiplication-for-layers-of-simple-feed-forward-anns/
            # https://www.google.com/search?q=numpy+batch+matrix+multiplication&oq=numpy+batch+&aqs=chrome.1.69i57j0i20i263i512j0i512j0i20i263i512j0i512l6.4406j0j7&sourceid=chrome&ie=UTF-8

            # activations = layers_activation_funcs[layer_index](
            #     np.dot(activations, layer_weights) + layer_biases
            # )

        return activations

    feedforward(training_data)

    def backprop(x, y):
        pass

    def SGD():
        for epoch in range(epochs):
            np.random.shuffle(training_data)
            start_time = perf_counter()

            for batch in [
                training_data[batch_index : batch_index + batch_size]
                for batch_index in range(0, len(training_data), batch_size)
            ]:
                pass

            # on_epoch_end(epoch, perf_counter() - start_time, model)

    SGD()


FCNN(
    np.array([[1, 2, 3], [4, 5, 6]]),
    np.array([1, 0]),
    [10],
    [
        lambda x: 1 / (1 + np.exp(-x)),
    ],
    [
        lambda x: x * (1 - x),
    ],
    batch_size=3,
    epochs=1,
    seed=1,
)
