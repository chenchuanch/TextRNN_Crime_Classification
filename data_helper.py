import os
import re
import logging
import itertools
#import tensorflow as tf
import pandas as pd
import numpy as np
import json as js
from collections import Counter

logging.getLogger().setLevel(logging.INFO)

DEBUG = True

def clean_str(s):
	s = re.sub(r"[^A-Za-z0-9:(),!?\'\`]", " ", s)
	s = re.sub(r" : ", ":", s)
	s = re.sub(r"\'s", " \'s", s)
	s = re.sub(r"\'ve", " \'ve", s)
	s = re.sub(r"n\'t", " n\'t", s)
	s = re.sub(r"\'re", " \'re", s)
	s = re.sub(r"\'d", " \'d", s)
	s = re.sub(r"\'ll", " \'ll", s)
	s = re.sub(r",", " , ", s)
	s = re.sub(r"!", " ! ", s)
	s = re.sub(r"\(", " \( ", s)
	s = re.sub(r"\)", " \) ", s)
	s = re.sub(r"\s{2,}", " ", s)
	return s.strip().lower()

def pad_sentences(sentences, padding_word="<PAD/>", forced_sequence_lengh=None):
	logging.info('Padding Centences...')

	if forced_sequence_lengh is None: #train
		sequence_length = max(len(x) for x in sentences)
	else:#prediection
		sequence_length = forced_sequence_lengh
	logging.critical('The maximum length is {}'.format(sequence_length))

	padded_sentences = []
	for i in range(len(sentences)):
		sentence = sentences[i]
		num_padding = sequence_length - len(sentence)
		if num_padding < 0:
			logging.info('This sentence has to cut off because it is longer than trained sequence length')
			padded_sentence = sentence[0:sequence_length]
		else:
			padded_sentence = sentence + [padding_word] * num_padding
		padded_sentences.append(padded_sentence)
	return padded_sentences

def build_vocab(sentences):
	logging.info('Build vocabulary...')

	word_counts = Counter(itertools.chain(*sentences))
	vocabulary_inv = [word[0] for word in word_counts.most_common()]

def load_data(filename):
	logging.info('Loading Data...')

	df = pd.read_csv(filename, compression = 'zip')
	selected = ['Category', 'Descript']
	non_selected = list(set(df.columns) - set(selected))

	df = df.drop(non_selected, axis=1)
	df = df.dropna(axis=0, how='any', subset=selected)
	df = df.reindex(np.random.permutation(df.index))

	labels = sorted(list(set(df[selected[0]].tolist())))
	num_label = len(labels)
	one_hot = np.zeros((num_label,num_label),int)
	np.fill_diagonal(one_hot, 1)
	label_dict = dict(zip(labels, one_hot))

	x_raw = df[selected[1]].apply(lambda x: clean_str(x).split(' ')).tolist()
	y_raw = df[selected[0]].apply(lambda y: label_dict[y]).tolist()

	x_raw = pad_sentences(x_raw)
	vocabulary, vocabulary_inv = build_vocab(x_raw)

if __name__=="__main__":
	train_file = './data/train.csv.zip'
	load_data(train_file)