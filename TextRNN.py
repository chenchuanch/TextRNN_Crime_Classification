import tensorflow as tf
import numpy as np
import logging

logging.getLogger().setLevel(logging.INFO)

class TextRNN(object):
  def __init__(self, num_class, batch_size, sequence_length, embedding_size, hidden_size, vocab_size, initializer=tf.random_normal_initializer(stddev=0.1)):
    #decay_steps, decay_rate
    
    # set hyperparamter
    self.vocab_size = vocab_size
    self.hidden_size = hidden_size
    self.embedding_size = embedding_size
    self.num_class = num_class
    #self.learning_rate = learning_rate

    
    self.batch_size = batch_size
    self.initializer=initializer    

    # add placeholder (X,label)
    self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name = 'input_x')
    self.input_y = tf.placeholder(tf.int32, [None, num_class], name = 'input_y')
    self.dropout_keep_prob = tf.placeholder(tf.float32, name = 'dropout_keep_prob')
    
    self.global_step = tf.Variable(0, name='global_step', trainable=False)
    self.epoch_step = tf.Variable(0, name='epoch_step', trainable=False)
    self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))
    #self.decay_steps, self.decay_rate = decay_steps, decay_rate

    self.instantiate_weight()
    self.logits = self.inference()
    
    self.loss = self.loss()
    self.train_op = self.train()
    self.accuracy = self.accuracy()

    logging.info("num_class: %d\tbatch_size: %d\tsequence_length: %d\tembedding_size: %d\thidden_size: %d\tvocab_size: %d" %(num_class, batch_size, sequence_length, embedding_size, hidden_size, vocab_size))

  def instantiate_weight(self):
    with tf.name_scope("emmbedding"):
      self.Embedding = tf.get_variable("Embedding", shape=[self.vocab_size, self.embedding_size], initializer=self.initializer)
      self.W_projection = tf.get_variable("W_projection", shape=[self.hidden_size*2, self.num_class], initializer=self.initializer)
      self.b_projection = tf.get_variable("b_projection", shape=[self.num_class])

  def inference(self):
    # 1. get emebedding of words in the sentence
    self.embedded_words = tf.nn.embedding_lookup(self.Embedding, self.input_x)
    #logging.info("self.embedded_words: {}".format(self.embedded_words))

    # 2. Bi-lstm layer
    lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_size)
    lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_size)
    if self.dropout_keep_prob is not None:
      lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(lstm_fw_cell, output_keep_prob=self.dropout_keep_prob)
      lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(lstm_bw_cell, output_keep_prob=self.dropout_keep_prob)
    outputs,_ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, self.embedded_words, dtype=tf.float32)

    # 3. concat output
    output_rnn = tf.concat(outputs, axis=2)  #????
    self.output_rnn_last = output_rnn[:, -1, :]

    # 4, logits
    with tf.name_scope("output"):
      logits = tf.matmul(self.output_rnn_last, self.W_projection) + self.b_projection
    return logits

  def loss(self, l2_lambda=0.0001):
    with tf.name_scope("loss"):
      losses = tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.logits)
      l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
      loss = tf.reduce_mean(losses) + l2_losses
    return loss

  def accuracy(self):
    prediction = tf.argmax(self.logits, axis=1, name="predictions")
    correct_prediction = tf.equal(tf.cast(prediction, tf.int32), self.input_y)
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy")
    return acc

  def train(self):
    #learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps,self.decay_rate, staircase=True)
    #train_op = tf.contrib.layers.optimize_loss(self.loss, global_step=self.global_step, learning_rate=learning_rate, optimizer="Adam")
    optimizer = tf.train.RMSPropOptimizer(1e-3, decay=0.9)
    grads_and_vars = optimizer.compute_gradients(self.loss)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)
    return train_op
