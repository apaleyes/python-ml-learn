import tensorflow as tf
import pickle
import numpy as np
import os.path

n_nodes_hl1 = 500
n_nodes_hl2 = 500

n_classes = 2
n_epochs = 10

x = tf.placeholder('float')
y = tf.placeholder('float')

with open('lexicon.pickle', 'rb') as f:
	lexicon = pickle.load(f)

input_length = len(lexicon)

def neural_network_model(data):
    input_sizes = [input_length, n_nodes_hl1, n_nodes_hl2]
    nodes = [n_nodes_hl1, n_nodes_hl2, n_classes]
    layer_definitions = []
    for i, n in zip(input_sizes, nodes):
        layer_definition = {
            'weights': tf.Variable(tf.random_normal([i, n])),
            "biases": tf.Variable(tf.random_normal([n]))
        }
        layer_definitions.append(layer_definition)

    layer_input = data
    for layer_definition in layer_definitions[:-1]:
        layer = tf.add(tf.matmul(layer_input, layer_definition['weights']), layer_definition['biases'])
        layer = tf.nn.relu(layer)

        layer_input = layer

    output = tf.add(tf.matmul(layer_input, layer_definitions[-1]['weights']), layer_definitions[-1]['biases'])
    return output