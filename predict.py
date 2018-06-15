import os
import logging
import TextRNN
import numpy as np
import data_helper
import pandas as pd
import tensorflow as tf

logging.getLogger().setLevel(logging.INFO)

def load_test_data(test_file, labels):
  df = pd.read_csv(test_file, sep='|')
  select = ['Descript']

  df = df.dropna(axis=0, how='any', subset=select)
  test_examples = df[select[0]].apply(lambda x:data_helper.clean_str(x).split(' ')).tolist()

  num_labels = len(labels)
  ont_hot = np.zeros((num_labels, num_labels), int)
  np.fill_diagonals(one_hot, 1)
  label_dict = dict(zip(labels, ont_hot))

def prediction_data():
  trained_dir = sys.argv[1]
  if not trained_dir.endswith('/'):
    trained_dir += '/'
  test_file = sys.argv[2]
  