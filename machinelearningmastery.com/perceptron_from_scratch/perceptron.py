def predict(input_row, weights):
    """
    Activate perceptron and calculate the output value
    Last weight is assumed to be the bias not corresponding to any input
    """

    if len(input_row) + 1 != len(weights):
        raise ValueError("Input length should be {} (number of weights minus 1), received {}".format(len(weights) - 1, len(input_row)))

    bias = weights[-1]
    activation = sum([i * w for (i, w) in zip(input_row, weights[:-1])]) + bias
    output = 1.0 if activation >= 0.0 else 0.0
    return output

def train_weights(train_data, learning_rate, n_epoch):
    """
    Trains the perceptron on the training data
    Data is an list of training rows with last column being an output label
    """

    # -1 because train_data includes labels
    number_of_input_weights = len(train_data[0]) - 1
    # +1 for bias
    weights = [0.0] * (number_of_input_weights + 1)

    for epoch in range(n_epoch):
        sum_error = 0.0
        for row in train_data:
            prediction = predict(row[:-1], weights)
            expected = row[-1]
            error = expected - prediction
            sum_error += error ** 2

            # update bias
            weights[-1] = weights[-1] + learning_rate * error
            # update input weights
            for i in range(0, len(weights) - 1):
                weights[i] = weights[i] + learning_rate * error * row[i]

        print('>epoch={}, learning rate={:.3}, error={:.3}'.format(epoch, learning_rate, sum_error))

    return weights

def perceptron(train_data, test_data, learning_rate, n_epoch):
    predictions = []
    weights = train_weights(train_data, learning_rate, n_epoch)
    for row in test_data:
        prediction = predict(row, weights)
        predictions.append(prediction)
    return predictions