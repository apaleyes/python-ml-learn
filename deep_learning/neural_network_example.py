import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data", one_hot=True)

n_classes = 10
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500
batch_size = 100
image_pixels = 28 * 28

x = tf.placeholder('float', [None, image_pixels])
y = tf.placeholder('float')

def neural_network_model(data):
	input_sizes = [image_pixels, n_nodes_hl1, n_nodes_hl2, n_nodes_hl3]
	nodes = [n_nodes_hl1, n_nodes_hl2, n_nodes_hl3, n_classes]
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

sample_data = tf.reshape([0.0] * image_pixels, [-1, image_pixels])
sample_result = neural_network_model(sample_data)

with tf.Session() as session:
	print(session.run(sample_result))