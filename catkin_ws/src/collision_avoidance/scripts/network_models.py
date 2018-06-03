#!/usr/bin/env python
import tensorflow as tf 
import tensorflow.contrib.slim as slim

class DRQN:
  def __init__(self, state_size, action_size, learning_rate=0.01, hidden_size=10, rnn_cell=None, name='QNetwork'):
    with tf.variable_scope(name):
      with tf.name_scope("Prediction"):       
        # Placeholders for the input image, the trace length and batch size of the RNN block, and keep probabilty of the Dropout layers
        self.image_input = tf.placeholder(dtype=tf.float32, shape=[None, state_size], name='inputs')
        self.rnn_trace_length = tf.placeholder(dtype=tf.int32)
        self.rnn_batch_size = tf.placeholder(dtype=tf.int32, shape=[])
        self.keep_per = tf.placeholder(shape=None, dtype=tf.float32)
        
        # Get size of sqaure picture (divide by 3 because of 3 channels in image i.e. because image is RGB)
        pixels = int((state_size//3) ** 0.5)
        reshaped_input = tf.reshape(self.image_input, shape=[-1, pixels, pixels, 3])
        # Pass image through 4 convolutional layers into the RNN block
        conv = slim.conv2d(inputs=reshaped_input, num_outputs=32, kernel_size=[8,8], stride=[4,4], padding='VALID', biases_initializer=None)
        conv = slim.conv2d(inputs=conv, num_outputs=64, kernel_size=[4,4], stride=[2,2], padding='VALID', biases_initializer=None)
        conv = slim.conv2d(inputs=conv, num_outputs=64, kernel_size=[3,3], stride=[1,1], padding='VALID', biases_initializer=None)
        conv = slim.conv2d(inputs=conv, num_outputs=hidden_size, kernel_size=[7,7], stride=[1,1], padding='VALID', biases_initializer=None)

        self.rnn_state_in = rnn_cell.zero_state(self.rnn_batch_size, tf.float32)
        self.rnn, self.rnn_state = tf.nn.dynamic_rnn(inputs=tf.reshape(slim.flatten(conv), [self.rnn_batch_size, self.rnn_trace_length, hidden_size]), cell=rnn_cell, dtype=tf.float32, initial_state=self.rnn_state_in, scope='rnn')
        self.rnn = tf.reshape(self.rnn, shape=[-1, hidden_size])

        # Dueling architecture, where the output from teh RNN block is split into advantage and value streams
        advantage_stream = tf.contrib.layers.fully_connected(self.rnn, hidden_size, activation_fn=tf.nn.relu, scope='_fc_advantage_hidden')
        advantage_stream = tf.contrib.layers.fully_connected(advantage_stream, action_size, activation_fn=None, scope='_fc_advantage')
        # Each stream has a dropout layer at the end to simulate a Bayesian Neural Network, used for effective exploration of teh state space
        advantage_stream = slim.dropout(advantage_stream, self.keep_per)
        value_stream = tf.contrib.layers.fully_connected(self.rnn, hidden_size, activation_fn=tf.nn.relu, scope='_fc_value_hidden')
        value_stream = tf.contrib.layers.fully_connected(value_stream, 1, activation_fn=None, scope='_fc_value')          
        value_stream = slim.dropout(value_stream, self.keep_per)

        # Both streams are concatenated to produce the final Q values.
        self.output = value_stream + tf.subtract(advantage_stream, tf.reduce_mean(advantage_stream, axis=1, keep_dims=True))    

      with tf.name_scope('Training'):
        # During training, we get the actions taken and the target Q values from the target QN
        self.actions = tf.placeholder(tf.int32, [None], name='actions')
        self.targetQs = tf.placeholder(tf.float32, [None], name='target')
        
        # These values are used to get the actual Q values 
        actualQs = tf.reduce_sum(tf.multiply(self.output, tf.one_hot(self.actions, action_size)), axis=1)
        # Instead of sending all the gradients back during backprop, we only send the second half (based on a Carnegie Mellon paper)
        # Done by zeroing out the first half
        mask_first_half = tf.zeros([self.rnn_batch_size, self.rnn_trace_length//2])
        mask_second_half = tf.ones([self.rnn_batch_size, self.rnn_trace_length//2])
        # Loss is the squared difference between target Qs and actual Qs for the second half of the batch
        self.loss = tf.reduce_mean(tf.square(self.targetQs - actualQs) * tf.reshape(tf.concat([mask_first_half, mask_second_half], 1), [-1]))
        # This loss is minimized during training
        self.opt = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

  def get_Q_values(self, sess, state, trace_length, rnn_state_in, batch_size, keep_per):
    return sess.run(self.output, \
      feed_dict={self.image_input: state, self.rnn_trace_length: trace_length, \
      self.rnn_state_in: rnn_state_in, self.rnn_batch_size: batch_size, self.keep_per: keep_per})
  
  def update_network(self, sess, states, targets, actions, trace_length, reccurent_state_train, batch_size, keep_per ):
    loss, _ = sess.run([self.loss, self.opt], \
        feed_dict={self.image_input: states, self.targetQs: targets,\
        self.actions: actions, self.rnn_trace_length: trace_length,\
        self.rnn_state_in: reccurent_state_train, self.rnn_batch_size: batch_size, self.keep_per: keep_per})
    return loss

  def rnn_hidden_state(self, sess, state, trace_length, rnn_state_in, batch_size):
    return sess.run(self.rnn_state, \
      feed_dict={self.image_input: state, self.rnn_trace_length: trace_length, \
      self.rnn_state_in: rnn_state_in, self.rnn_batch_size: batch_size})