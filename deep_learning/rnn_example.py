import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data", one_hot=True)

n_epochs = 10
n_classes = 10

batch_size = 128
chunk_size = 28
n_chunks = 28
rnn_size = 128


x = tf.placeholder('float', [None, n_chunks, chunk_size])
y = tf.placeholder('float')

def recurrent_neural_network(x):
	layer = {
				"weights": tf.Variable(tf.random_normal([rnn_size, n_classes])),
				"biases": tf.Variable(tf.random_normal([n_classes]))
			}

	x = tf.transpose(x, [1,0,2])
	x = tf.reshape(x, [-1, chunk_size])
	x = tf.split(0, n_chunks, x)

	lstm_cell = rnn_cell.BasicLSTMCell(rnn_size, state_is_tuple=True)
	outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)

	output = tf.add(tf.matmul(outputs[-1], layer['weights']), layer['biases'])
	return output

def train_neural_network(x, y):
	prediction = recurrent_neural_network(x)
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
				epoch_x = epoch_x.reshape((batch_size, n_chunks, chunk_size))
				_, c = session.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
				epoch_loss += c
			print('Epoch', epoch, 'completed out of', n_epochs, ". Loss:", epoch_loss)

		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		print('Accuracy:', accuracy.eval({x: mnist.test.images.reshape((-1, n_chunks, chunk_size)), y: mnist.test.labels}))

train_neural_network(x, y)

# Simple sanity testing code
# sample_data = tf.reshape([0.0] * image_pixels, [-1, image_pixels])
# sample_result = neural_network_model(sample_data)

# with tf.Session() as session:
# 	session.run(tf.global_variables_initializer())
# 	print(session.run(sample_result))