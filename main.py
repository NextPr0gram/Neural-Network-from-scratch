import numpy as np

""" 
inputs = [1, 2, 3, 2.5]

weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2, 3, 0.5]

layer_outputs = []

for neuron_weights, neuron_bias in zip(weights, biases): # [(weight1, bias1), (weight2, bias2), (weight3, bias3)]
    neuron_output = 0
    for n_input, weight in zip(inputs, neuron_weights): # [(input1, n_w_1), (input2, n_w_2), (input3, n_w_3), (input4, n_w_4)]
        neuron_output += n_input*weight
    neuron_output += neuron_bias
    layer_outputs.append(neuron_output)

print(layer_outputs) """

# dot product using numpy
inputs = [1, 2, 3, 2.5]
weights = [0.2, 0.8, -0.5, 1.0]
biases = [2, 3, 0.5]

output = np.dot(weights, inputs) + biases
print(output)