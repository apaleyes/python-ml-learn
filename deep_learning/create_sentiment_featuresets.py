import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import random
import pickle
from collections import Counter

lemmatizer = WordNetLemmatizer()
n_lines = 1000000

def create_lexicon(files):
	full_lexicon = []

	for file in files:
		with open(file, 'r') as f:
			contents = f.readlines()

		for line in contents[:n_lines]:
			all_words = word_tokenize(line.lower())
			full_lexicon += list(all_words)

	full_lexicon = [lemmatizer.lemmatize(w) for w in full_lexicon]
	w_counts = Counter(full_lexicon)

	short_lexicon = []
	for w in w_counts:
		if 1000 > w_counts[w] > 50:
			short_lexicon.append(w)

	print("Lexicon size", len(short_lexicon))
	return short_lexicon

def sample_handling(sample, lexicon, classification):
	featureset = []
	with open(sample, 'r') as f:
		contents = f.readlines()

	for line in contents[:n_lines]:
		current_words = word_tokenize(line.lower())
		current_words = [lemmatizer.lemmatize(w) for w in current_words]
		features = [0] * len(lexicon)
		for word in current_words:
			if word.lower() in lexicon:
				i = lexicon.index(word.lower())
				features[i] += 1

		features = list(features)
		featureset.append([features, classification])

	return featureset

def create_features_and_labels(pos, neg, test_size=0.1):
	lexicon = create_lexicon([pos, neg])
	features = []
	features += sample_handling(pos, lexicon, [1,0])
	features += sample_handling(neg, lexicon, [0,1])
	random.shuffle(features)
	features = np.array(features)

	testing_size = int(test_size * len(features))
	train_x = list(features[:, 0][:-testing_size])
	train_y = list(features[:, 1][:-testing_size])
	test_x = list(features[:, 0][-testing_size:])
	test_y = list(features[:, 1][-testing_size:])

	return train_x, train_y, test_x, test_y

if __name__ == "__main__":
	train_x, train_y, test_x, test_y = create_features_and_labels("pos.txt", "neg.txt")
	with open("sentiment_set.pickle", "wb") as f:
		pickle.dump([train_x, train_y, test_x, test_y], f)