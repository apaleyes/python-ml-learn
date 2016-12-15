import tensorflow as tf
import pickle
import numpy as np
import os.path
from sentiment_big_dataset_preprocessing import vectorize_line

n_nodes_hl1 = 500
n_nodes_hl2 = 500

n_classes = 2
n_epochs = 5
batch_size = 32

x = tf.placeholder('float')
y = tf.placeholder('float')

with open('lexicon.pickle', 'rb') as f:
	lexicon = pickle.load(f)

input_length = len(lexicon)
tf_log = "tf.log"
checkpoint_path = "./model.ckpt"

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

def train_neural_network(x, y):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction, y) )
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

    tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        if os.path.exists(tf_log):
            epoch = int(open(tf_log, 'r').readlines()[-1]) + 1
        else:
            epoch = 1

        while epoch <= n_epochs:
            print('Starting epoch ', epoch)
            if epoch != 1:
                saver.restore(session, checkpoint_path)
            epoch_loss = 1

            with open("train_set_shuffled.txt", buffering=200000, encoding="latin-1") as f:
                batch_x = []
                batch_y = []
                batches_run = 0
                for line in f:
                    line_x, line_y = vectorize_line(line, lexicon)
                    batch_x.append(line_x)
                    batch_y.append(line_y)
                    if len(batch_x) >= batch_size:
                        _, c = session.run([optimizer, cost], feed_dict={x: np.array(batch_x), y: np.array(batch_y)})
                        epoch_loss += c
                        batch_x = []
                        batch_y = []
                        batches_run += 1

                        if (batches_run/1000).is_integer():
                            print('Batch run ', batches_run, ". Epoch loss ", epoch_loss)

            saver.save(session, checkpoint_path)
            print("Epoch ", epoch, "completed out of ", n_epochs)
            with open(tf_log, 'a') as f:
                f.write(str(epoch))

            epoch += 1


def test_neural_network():
    x = tf.placeholder('float')
    y = tf.placeholder('float')
    prediction = neural_network_model(x)
    
    tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        saver.restore(session, checkpoint_path)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        feature_sets = []
        labels = []
        with open('vectorized_test_data.txt') as f:
            for line in f:
                features = list(eval(line.split('::')[0]))
                label = list(eval(line.split('::')[1]))

                feature_sets.append(features)
                labels.append(label)

        test_x = np.array(feature_sets)
        test_y = np.array(labels)
        print("Accuracy:", accuracy.eval({x: test_x, y: test_y}))

train_neural_network(x, y)
test_neural_network()

