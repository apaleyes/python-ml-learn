import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data", one_hot=True)

n_classes = 10
batch_size = 128
input_size = 28*28

x = tf.placeholder('float', [None, input_size])
y = tf.placeholder('float')

def conv2d(x, w):
	return tf.nn.conv2d(x, w, strides=[1,1,1,1], padding="SAME")

def maxpool2d(x):
	return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")

def convolutional_neural_network_model(x):
	weights = {
		'w_conv1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
		'w_conv2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
		'w_fc': tf.Variable(tf.random_normal([7*7*64, 1024])),
		'out': tf.Variable(tf.random_normal([1024, n_classes]))
	}

	biases = {
		'b_conv1': tf.Variable(tf.random_normal([32])),
		'b_conv2': tf.Variable(tf.random_normal([64])),
		'b_fc': tf.Variable(tf.random_normal([1024])),
		'out': tf.Variable(tf.random_normal([n_classes]))
	}

	x = tf.reshape(x, shape=[-1, 28, 28, 1])

	conv1 = tf.nn.relu(conv2d(x, weights['w_conv1']) + biases['b_conv1'])
	conv1 = maxpool2d(conv1)
	
	conv2 = tf.nn.relu(conv2d(conv1, weights['w_conv2']) + biases['b_conv2'])
	conv2 = maxpool2d(conv2)

	fc = tf.reshape(conv2, shape=[-1, 7*7*64])
	fc = tf.nn.relu(tf.matmul(fc, weights['w_fc']) + biases['b_fc'])

	output = tf.matmul(fc, weights['out']) + biases['out']

	return output

def train_neural_network(x, y):
	prediction = convolutional_neural_network_model(x)
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