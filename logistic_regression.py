import numpy as np


def logistic_regression(samples, targets, alpha=0.01):
    weights = np.random.randn(samples.shape[1])
    bias = np.random.randn()

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def feedforward(x):
        weighted_sum = x.dot(weights) + bias
        return sigmoid(weighted_sum)

    while True:
        # forward propagation
        outputs = feedforward(samples)

        # backward propagation
        d_loss_by_d_output = outputs - targets
        d_loss_by_d_weights = samples.T.dot(d_loss_by_d_output)
        d_loss_by_d_bias = d_loss_by_d_output.sum()

        # update
        new_weights = weights - alpha * d_loss_by_d_weights
        new_bias = bias - alpha * d_loss_by_d_bias

        if np.allclose([*weights, bias], [*new_weights, new_bias]):
            return feedforward

        weights = new_weights
        bias = new_bias
