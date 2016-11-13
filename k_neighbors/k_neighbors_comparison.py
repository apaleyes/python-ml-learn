import numpy as np
import warnings
from collections import Counter
import pandas as pd
import random
from sklearn import preprocessing, model_selection, neighbors
import time

def k_neighbors(data, predict, k=3):
	if len(data) >= k:
		warnings.warn('K is less than the number of classes')

	distances = []
	for group in data:
		for features in data[group]:
			distances.append((np.linalg.norm(np.array(features) - np.array(predict)), group))
	votes = [d[1] for d in sorted(distances)[:k]]
	vote_result = Counter(votes).most_common(1)[0][0]
	confidence = Counter(votes).most_common(1)[0][1] / k

	return vote_result, confidence

def prepare_data():
	df = pd.read_csv("breast-cancer-wisconsin.data")
	df.replace('?', -99999, inplace=True)
	df.drop(['id'], 1, inplace=True)
	return df

def run_homebrew(df, n, test_size):
	accuracies = []
	for i in range(n):
		full_data = df.astype(float).values.tolist()

		random.shuffle(full_data)
		
		train_set = {2:[], 4:[]}
		test_set = {2:[], 4:[]}
		train_data = full_data[:-int(test_size*len(full_data))]
		test_data = full_data[-int(test_size*len(full_data)):]

		for i in train_data:
			train_set[i[-1]].append(i[:-1])
		for i in test_data:
			test_set[i[-1]].append(i[:-1])

		correct = 0
		total = 0
		for group in test_set:
			for data in test_set[group]:
				vote, confidence = k_neighbors(train_set, data, k=5)
				if group == vote:
					correct += 1
				total += 1

		accuracies.append(correct/total)

	return np.mean(accuracies)

def run_scikit(df, n, test_size):
	accuracies = []
	for i in range(n):
		X = np.array(df.drop(['class'], 1))
		y = np.array(df['class'])

		X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=test_size)

		clf = neighbors.KNeighborsClassifier(algorithm='brute')
		clf.fit(X_train, y_train)
		accuracy = clf.score(X_test, y_test)
		accuracies.append(accuracy)

	return np.mean(accuracies)

def measure(func, name, df):
	start = time.clock()
	accuracy = func(df) * 100.0
	end = time.clock()
	print("{} accuracy: {} %, time: {} s".format(name, accuracy, end - start))


df = prepare_data()
measure(lambda df: run_homebrew(df, 5, 0.2), "Homebrew", df)
measure(lambda df: run_scikit(df, 5, 0.2), "Scikit-learn", df)