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





