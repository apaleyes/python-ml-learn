import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
import pandas as pd
import random

lemmatizer = WordNetLemmatizer()

'''
polarity 0 = negative. 2 = neutral. 4 = positive.
id
date
query
user
tweet
'''

def format_input(fin, fout):
	outfile = open(fout, 'a')
	with open(fin, buffering=200000, encoding="latin-1") as f:
		for line in f:
			try:
				line = line.replace('"', '')
				polarity = line.split(',')[0]
				if polarity == '0':
					polarity = [1,0]
				elif polarity == '4':
					polarity = [0,1]
				else:
					polarity = [0,0]

				tweet = line.split(',')[-1].lower()
				outline = "{}:::{}".format(str(polarity), tweet)
				outfile.write(outline)
			except Exception as e:
				pass
	outfile.close()

	print("Formatted {}".format(fin))


def create_lexicon(fin, lexicon_pickle):
	lexicon = []
	lines = []
	with open(fin, buffering=200000, encoding="latin-1") as f:
		try:
			i = 0
			for line in f:
				i += 1
				if (i / 2500).is_integer():
					tweet = line.split(':::')[1]
					lines.append(tweet)
		except Exception as e:
			print(str(e))

	all_lines = ' '.join(lines)
	tokens = word_tokenize(all_lines)
	lexicon = list(set([lemmatizer.lemmatize(i) for i in tokens]))

	with open(lexicon_pickle, 'wb') as f:
		pickle.dump(lexicon, f)

	print("Created lexicon")


def vectorize_line(line, lexicon):
	label = line.split(":::")[0]
	tweet = line.split(":::")[1]
	current_tokens = word_tokenize(tweet)
	current_words = [lemmatizer.lemmatize(t) for t in current_tokens]

	features = np.zeros(len(lexicon))

	for word in current_words:
		if word in lexicon:
			index_value = lexicon.index(word)
			# or try += 1 here
			features[index_value] = 1

	features = list(features)
	labels = eval(label)

	return features, labels


def convert_data_to_vector(fin, lexicon_pickle, fout):
	with open(lexicon_pickle, 'rb') as f:
		lexicon = pickle.load(f)

	outfile = open(fout, 'a')
	with open(fin, buffering=200000, encoding="latin-1") as f:
		i = 0
		for line in f:
			i += 1
			features, labels = vectorize_line(line, lexicon)
			outline = "{}::{}\n".format(str(features), str(labels))
			outfile.write(outline)


def shuffle_data(fin, fout):
	lines = open(fin).readlines()
	random.shuffle(lines)
	open(fout, 'w').writelines(lines)


if __name__ == '__main__':

	format_input('training.1600000.processed.noemoticon.csv','train_set.txt')
	format_input('testdata.manual.2009.06.14.csv','test_set.txt')

	create_lexicon('train_set.txt', 'lexicon.pickle')

	convert_data_to_vector('test_set.txt', 'lexicon.pickle', 'vectorized_test_data.txt')

	shuffle_data('train_set.txt', 'train_set_shuffled.txt')