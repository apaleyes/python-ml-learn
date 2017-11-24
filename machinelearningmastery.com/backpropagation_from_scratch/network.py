from random import random

def initialize_network(n_inputs, n_hidden, n_ouputs):
    hidden_layer = [{'weights': [random() for _ in range(n_inputs + 1)]} for _ in range(n_hidden)]
    output_layer = [{'weights': [random() for _ in range(n_hidden + 1)]} for _ in range(n_ouputs)]
    network = [hidden_layer, output_layer]
    return network

