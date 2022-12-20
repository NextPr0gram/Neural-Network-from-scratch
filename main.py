from Layer_Dense import Layer_Dense
from nnfs.datasets import spiral_data
from Activation_ReLu import Activation_ReLu

X, y = spiral_data(100, 3)  # data set


layer1 = Layer_Dense(2, 5)
activation1 = Activation_ReLu()

layer1.forward(X)
activation1.forward(layer1.output)
print(activation1.output)

