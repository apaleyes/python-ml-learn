import tensorflow as tf
import numpy as np
from create_sentiment_featuresets import create_features_and_labels

train_x,train_y,test_x,test_y = create_features_and_labels('pos.txt', 'neg.txt')

n_classes = 2
n_nodes_hl1 = 1500
n_nodes_hl2 = 1500
n_nodes_hl3 = 1500
batch_size = 100
input_length = len(train_x[0])

x = tf.placeholder('float', [None, len(train_x[0])])
y = tf.placeholder('float')

def neural_network_model(data):
    input_sizes = [input_length, n_nodes_hl1, n_nodes_hl2, n_nodes_hl3]
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
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    hm_epochs = 10
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            i = 0
            while i < len(train_x):
                start = i
                end = start + batch_size
                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
                epoch_loss += c
                i += batch_size

            print('Epoch', epoch+1, 'completed out of',hm_epochs,'loss:',epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x:test_x, y:test_y}))

train_neural_network(x, y)