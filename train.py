import os
import sys
import time
import logging
from TextRNN import TextRNN
import json as js
import numpy as np
import data_helper
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

logging.getLogger().setLevel(logging.INFO)

def trainTextRNN():
  input_file = sys.argv[1]
  x_, y_, vocabulary, vocabulary_inv, df, labels = data_helper.load_data(input_file)
  vocabulary_size = len(vocabulary)  

  training_config = sys.argv[2]
  params = js.loads(open(training_config).read())
  print('params: {}'.format(params))
  #split the dataset into train set and test set
  x, x_test, y, y_test = train_test_split(x_, y_, test_size=0.2)

  #split the train set into train set and dev set
  x_train, x_dev, y_train, y_dev = train_test_split(x, y, test_size=0.1)

  logging.info('x_train: {}, x_dev: {}, x_test: {}'.format(len(x_train), len(x_dev), len(x_test)))
  logging.info('y_train: {}, y_dev: {}, y_test: {}'.format(len(y_train), len(y_dev), len(y_test)))

  timestamp = str(int(time.time()))
  trained_dir = './trained_result_' + timestamp + '/'
  if os.path.exists(trained_dir):
    shutil.rmtree(trained_dir)
  os.makedirs(trained_dir)

  graph = tf.Graph()

  with graph.as_default():
    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_conf.gpu_options.allow_growth = True
    sess = tf.Session(config=session_conf)
    with sess.as_default():
      textRNN_mdoel = TextRNN(num_class=y_train.shape[1],batch_size=params['batch_size'],sequence_length=x_train.shape[1],embedding_size=params['embedding_dim'],hidden_size=params['hidden_unit'],vocab_size=vocabulary_size)
      
      saver = tf.train.Saver()
      checkpoint_dir = './checkpoint_' + timestamp + '/'
      if os.path.exists(checkpoint_dir):
        #shitil.rmtree(checkpoint_dir)
        logging.info('Restore Variable form checkpoint for rnn model')
        saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))
      else:
        logging.info('Initializing Varivales')
        sess.run(tf.global_variables_initializer())

      # feed data and training
      curr_epoch = sess.run(textRNN_mdoel.epoch_step)
      batch_size = params['batch_size']
      number_of_training_data = len(x_train)
      number_of_valuation_data = len(x_dev)
      number_of_test_data = len(x_test)

      for epoch in range(curr_epoch, params['num_epochs']):
        loss, acc, counter = 0.0, 0.0, 0

        for start, end in zip(range(0, number_of_training_data, batch_size), range(batch_size, number_of_training_data, batch_size)):
          
          data = list(zip(x_train, y_train))
          data = np.array(data)
          x_data, y_data = zip(*data)

          logging.info("===============")
          logging.info(type(y_data))

          if epoch == 0 and counter == 0:
            #logging.info('x_train[{}:{}]: {}'.format(start, end, x_data))
            #logging.info('y_train[{}:{}]: {}'.format(start, end, y_data))
            logging.info('y_train shape: {}'.format(len(y_data)))
          print("begin sess run............")
          curr_loss, curr_acc, _= sess.run([textRNN_mdoel.loss, textRNN_mdoel.accuracy, textRNN_mdoel.train_op], feed_dict={
            textRNN_mdoel.input_x: x_data, 
            textRNN_mdoel.input_y: y_data,
            textRNN_mdoel.dropout_keep_prob:params['dropout_keep_prob'],
            })
          print("one batch passed............")
          loss, counter, acc = loss + curr_loss, counter + 1, acc + curr_acc

          if counter % 500 == 0:
            logging.info("Epoch %d\t Batch %d\tTrain Loss: %.3f\tTrain Accuracy" %(epoch, counter, loss/float(counter), acc/float(acc)))

        logging.info('Going to increment epoch counter...')
        sess.run(textRNN_mdoel.epoch_increment)

        # validation
        logging.info('Going to validation...')
        if epoch % params['valuation_every'] == 0:
          eva_loss, eva_acc, eva_counter = 0.0, 0.0, 0
          for start, end in zip(range(0, number_of_valuation_data, batch_size), range(batch_size, number_of_valuation_data, batch_size)):
            curr_eva_loss, logits, curr_eva_acc = sess.run([textRNN_mdoel.loss, textRNN_mdoel.logits, textRNN_mdoel.accuracy], feed_dict={
              textRNN_mdoel.input_x:x_dev[start:end],
              textRNN_mdoel.input_y:y_dev[start:end],
              textRNN_mdoel.dropout_keep_prob:1
              })
            eva_loss, eva_acc, eva_counter = eva_loss + curr_eva_loss, eva_acc + curr_eva_loss, eva_counter + 1
          eva_loss, eva_acc = eva_loss/float(eva_counter), eva_acc/float(eva_counter)
      
      #test
      test_loss, test_acc, test_counter = 0.0, 0.0, 0
      for start, end in zip(range(0, number_of_test_data, batch_size), range(batch_size, number_of_test_data, batch_size)):
        curr_test_loss, logits, curr_test_acc = sess.run([textRNN_mdoel.loss, textRNN_mdoel.logits, textRNN_mdoel.accuracy], feed_dict={
          textRNN_mdoel.input_x:x_dev[start:end],
          textRNN_mdoel.input_y:y_dev[start:end],
          textRNN_mdoel.dropout_keep_prob:1
          })
        test_loss, test_acc, test_counter = test_loss + curr_test_loss, test_acc + curr_test_loss, test_counter + 1
      eva_loss, eva_acc = eva_loss/float(eva_counter), eva_acc/float(eva_counter)

if __name__ == '__main__':
  trainTextRNN()
