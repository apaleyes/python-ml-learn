from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf

input_size = 28*28
n_classes = 10

x = tf.placeholder(tf.float32, [None, input_size])
W = tf.Variable(tf.zeros([input_size, n_classes]))
b = tf.Variable(tf.zeros([n_classes]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, [None, n_classes])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as session:
	session.run(init)
	for i in range(1000):
		batch_xs, batch_ys = mnist.train.next_batch(100)
		session.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

	print(session.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))