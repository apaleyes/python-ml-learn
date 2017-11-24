import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.estimator import regression
import tflearn.datasets.mnist as mnist

import numpy as np

X, Y, test_x, test_y = mnist.load_data(one_hot=True)

X = X.reshape([-1, 28, 28, 1])
test_x = test_x.reshape([-1, 28, 28, 1])

# input layer
convnet = input_data(shape=[None, 28, 28, 1], name='input')

# first convolutional layer
convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

# second convolutional layer
convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

# fully connected layer
convnet = fully_connected(convnet, 1024, activation='relu')

# output layer
convnet = fully_connected(convnet, 10, activation='softmax')

# optimizer
convnet = regression(convnet, optimizer='adam', learning_rate=0.001,
	loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet)

model.fit({'input': X}, {'targets': Y}, n_epoch=10,
	validation_set=({'input': test_x}, {'targets': test_y}),
	snapshot_step=500, show_metric=True, run_id='mnist')



prediction = model.predict(test_x)
correct = 0
for py, y in zip(prediction, test_y):
	if np.argmax(py) == np.argmax(y):
		correct += 1

accuracy = correct / len(test_y)
print("Accuracy:", accuracy)