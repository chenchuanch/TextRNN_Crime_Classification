import os
import re
import logging
#import tensorflow as tf
import pandas as pd
import numpy as np
import json as js

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
	if forced_sequence_lengh is None:
		sequence_lengh = max(len(x) for x in senteces)
	else:
		sequence_lengh = forced_sequence_lengh
	logging.critical('The maximum length is {}'.format(sequence_lengh))


def load_data(filename):
	df = pd.read_csv(filename, compression = 'zip')
	selected = ['Category', 'Descript']
	non_selected = list(set(df.columns) - set(selected))

	if DEBUG:
		print df.columns
		print "========"
		print set(df[selected[0]])

	df = df.drop(non_selected, axis=1)
	df = df.dropna(axis=0, how='any', subset=selected)
	df = df.reindex(np.random.permutation(df.index))

	labels = sorted(list(set(df[selected[0]].tolist())))
	num_label = len(labels)
	one_hot = np.zeros((num_label,num_label),int)
	np.fill_diagonal(one_hot, 1)
	label_dict = dict(zip(labels, one_hot))

	x_raw = df[selected[1]].apply(lambda x: clean_str(x).split(' ')).tolist()
	y_raw = df[selected[0]].apply(lambd y: label_dict[y]).tolist()

	x_raw = pad_sentences(x_raw)


	if DEBUG:
		print "========"
		print dict(zip(labels, one_hot))
		string = "aaa bbb ccc"
		print clean_str(string).split(' ')


	

if __name__=="__main__":
	train_file = './data/train.csv.zip'
	load_data(train_file)