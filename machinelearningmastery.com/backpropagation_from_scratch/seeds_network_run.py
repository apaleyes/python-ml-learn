from random import seed

from network import *
from dataset_operations import *

seed(1)

filename = 'seeds_dataset.csv'
dataset = load_csv(filename)
convert_to_float(dataset)
convert_to_int(dataset)
normalize_dataset(dataset)

n_folds = 5
l_rate = 0.3
n_epoch = 500
n_hidden = 5
scores = evaluate_algorithm(dataset, back_propagation, n_folds, l_rate, n_epoch, n_hidden)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))