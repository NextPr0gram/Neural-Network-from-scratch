import numpy as np

np.random.seed(0)


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1*np.random.randn(n_inputs, n_neurons)  # creates a matrix with random weights, n_inputs x n_neurons
        self.biases = np.zeros((1, n_neurons))  # 1 x n of neurons

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
