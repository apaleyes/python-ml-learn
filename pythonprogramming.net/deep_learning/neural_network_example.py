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

def train_neural_network(x, y):
	prediction = neural_network_model(x)
	cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction, y) )
	optimizer = tf.train.AdamOptimizer().minimize(cost)

	n_epochs = 10
	n_batches = int(mnist.train.num_examples / batch_size)
	with tf.Session() as session:
		session.run(tf.global_variables_initializer())

		for epoch in range(n_epochs):
			epoch_loss = 0.0
			for _ in range(n_batches):
				epoch_x, epoch_y = mnist.train.next_batch(batch_size)
				_, c = session.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
				epoch_loss += c
			print('Epoch', epoch, 'completed out of', n_epochs, ". Loss:", epoch_loss)

		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		print('Accuracy:', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

train_neural_network(x, y)

# Simple sanity testing code
# sample_data = tf.reshape([0.0] * image_pixels, [-1, image_pixels])
# sample_result = neural_network_model(sample_data)

# with tf.Session() as session:
# 	session.run(tf.global_variables_initializer())
# 	print(session.run(sample_result))