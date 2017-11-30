from random import randrange

def load_csv(filename):
    '''
    Reads dataset csv and stores it as list
    Assumes arbitrary number of spaces as a delimiter
    '''

    data = []
    with open(filename, 'r') as datafile:
        for row in datafile:
            if not row:
                continue
            data.append(row.split())

    return data

def convert_to_float(data):
    '''
    Converts all columns except last in the dataset into floats.
    Works in place, altering the given dataset
    '''
    for i, _ in enumerate(data):
        new_row = [float(x) for x in data[i][:-1]] + [data[i][-1]]
        data[i] = new_row

def convert_to_int(data):
    ''' Converts last column to int '''
    class_values = [row[-1] for row in data]
    unique = set(class_values)
    lookup = {}
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in data:
        row[-1] = lookup[row[-1]]

def dataset_minmax(data):
    '''
    Calculates min and max for given dataset
    Assumes last column as label as so skips it
    '''
    transposed = map(list, zip(*data))
    minmaxes = []
    for column in transposed[:-1]:
        column_minmax = {
            'min': min(column),
            'max': max(column)
        }
        minmaxes.append(column_minmax)
    return minmaxes

def normalize_dataset(data):
    '''
    Normalizes all dataset columns except last one to 0..1 range
    '''
    minmaxes = dataset_minmax(data)

    for row in data:
        for i, v in enumerate(row[:-1]):
            column_range = minmaxes[i]['max'] - minmaxes[i]['min']
            row[i] = (v - minmaxes[i]['min']) / column_range

def cross_validation_split(dataset, n_folds):
    ''' Split a dataset into n folds '''
    dataset_split = []
    dataset_copy = dataset[:]
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = []
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split

def accuracy_metric(actual, predicted):
    ''' Calculate accuracy percentage '''
    correct = 0
    for i in range(len(actual)):
        if int(round(actual[i])) == int(round(predicted[i])):
            correct += 1
    return correct / float(len(actual)) * 100.0

def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    ''' Evaluate an algorithm using a cross validation split '''
    folds = cross_validation_split(dataset, n_folds)
    scores = []
    for fold in folds:
        train_set = folds[:]
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = []
        for row in fold:
            row_copy = row[:]
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores