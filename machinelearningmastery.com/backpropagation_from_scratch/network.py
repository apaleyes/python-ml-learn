from random import random
import math

def initialize_network(n_inputs, n_hidden, n_ouputs):
    ''' Initializes a network is a single hidden layer '''
    hidden_layer = [{'weights': [random() for _ in range(n_inputs + 1)]} for _ in range(n_hidden)]
    output_layer = [{'weights': [random() for _ in range(n_hidden + 1)]} for _ in range(n_ouputs)]
    network = [hidden_layer, output_layer]
    return network

def activate(weights, inputs):
    '''
    Activation of a neuron - this is how neuron consumes its inputs given the weights
    Last weight is the bias, and so does not have a corresponding input value
    '''

    if len(weights) != len(inputs) + 1:
        message = "Dimension mismatch, " + \
                  "dimension of weights should equal to dimension of input plus 1 for bias. " + \
                  "Actual dimensions: weights {}, inputs {}".format(len(weights), len(inputs))
        raise ValueError(message)

    activation = 0
    for (w, i) in zip(weights[:-1], inputs):
        activation += w * i
    # add bias
    activation += weights[-1]

    return activation

def transfer(activation):
    '''
    Transfer of a neuron - this computes output of a neuron given its activation value
    There are different ways to do transfer, here Sigmoid function is used
    '''
    return 1.0 / (1.0 + math.exp(-activation))

def transfer_derivative(output):
    '''
    Derivative of the transfer given its output
    Derivative of sigmoid f(x) is f(x)*(1-f(x))
    '''
    return output * (1 - output)

def forward_propagate(network, input_row):
    ''' Forward propagation through a given network given a single row of input values '''
    layer_input = input_row
    for layer in network:
        layer_output = []
        for neuron in layer:
            activation = activate(neuron['weights'], layer_input)
            neuron['output'] = transfer(activation)
            layer_output.append(neuron['output'])
        layer_input = layer_output[:]

    return layer_output

def backward_propagate_error(network, expected):
    ''' Calculation of the error using backward propagation '''
    output_layer = network[-1]
    if len(expected) != len(output_layer):
        raise ValueError("Expected output length {} does not match output layer dimension {}".format(len(expected, len(output_layer))))

    for (i, neuron) in enumerate(output_layer):
        error = (expected[i] - neuron["output"]) * transfer_derivative(neuron["output"])
        neuron["delta"] = error

    for (i, layer) in reversed(list(enumerate(network))[:-1]):
        layer_index = i
        for (j, neuron) in enumerate(layer):
            error = 0
            for next_layer_neuron in network[layer_index + 1]:
                error += next_layer_neuron["delta"] * next_layer_neuron["weights"][j] * \
                         transfer_derivative(neuron["output"])
            neuron["delta"] = error

def update_weights(network, input_row, learning_rate):
    '''
    Update weights according to the input
    Assumes that the error was already calculated
    '''
    layer_input = input_row + [1.0]
    for layer in network:
        next_layer_input = []
        for (i, neuron) in enumerate(layer):
            new_weights = [w + learning_rate * neuron["delta"] * layer_input[i] for w in neuron["weights"]]
            neuron["weights"] = new_weights

            next_layer_input.append(neuron["output"])
        layer_input = next_layer_input + [1.0]



