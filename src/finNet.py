# ==============================================================================
# Author: Steve Schmidt
# Classes: FinNet()
# Abstract 2-D Image, 2 Hidden Layer  Convolutional Net with Max Pooling
# Intended use with financial ewma n day data, bollinger bands
# Potential use - any grayscale image classifier
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import time
import ipdb
import pandas as pd
import numpy as np
import csv
from sklearn.cross_validation import train_test_split

FLAGS = None

from logger import Logging
from reader import Reader


class FinNet(object):

  def __init__(self, params):
      self.start_time = time.time()
      self.params = params
      self.import_data(params)
      self.init_test_data(params)
      self.create_net(params)
      self.define_loss(params)
      self.init_net(params)
      self.nn_train(params)


#params must have matrix sizes for x, W, b, y
#filename to read
#

  def import_data(self, params):
      # Import data
      self.rdr = Reader()
      self.rdr.read_images('/home/ubuntu/store/fin_data', 'csv',True, False)
      self.xvals = np.array(self.rdr.c_images)
      self.yholder = {}
      with open('/home/ubuntu/store/fin_data/99999.y_vals.csv', 'r') as f:
        r = csv.reader(f)
        for row in r:
          self.yholder[int(row[0])] = float(row[1])
      self.yvals = [self.yholder[int(x)] for x in self.rdr.file_labels]
      self.yvals = np.array(self.yvals)
      self.xvals = pd.DataFrame(self.xvals)
      self.xvals['y'] = self.yvals
      #traindata, testdata = train_test_split(xvals, test_size=.25)
      self.traindata = self.xvals.loc[:1500]
      self.testdata = self.xvals.loc[1500:]
      print('read time: %s train size: %s test size: %s'%((time.time()-self.start_time),len(self.traindata),len(self.testdata)))

  def create_net(self, params):
      with tf.device('/cpu:0'):
          self.x = tf.placeholder(tf.float32, [None, 784])
          self.W = tf.Variable(tf.zeros([784, 1]))
          self.b = tf.Variable(tf.zeros([1]))
          self.y = tf.matmul(self.x, self.W) + self.b
  
  def define_loss(self, params):
      # Define loss and optimizer
      self.y_ = tf.placeholder(tf.float32, [None, 1])

  def init_net(self, params):
      with tf.device('/gpu:0'):
          # create the convolutional hidden layer sizes - make even more generic in future
          self.W_conv1 = self.weight_variable([5, 5, 1, 32])
          self.b_conv1 = self.bias_variable([32])
          self.x_image = tf.reshape(self.x, [-1,28,28,1])
          # relu
          self.h_conv1 = tf.nn.relu(self.conv2d(self.x_image, self.W_conv1) + self.b_conv1)
          # max pool
          self.h_pool1 = self.max_pool_2x2(self.h_conv1)
          # create second convolution layer
          self.W_conv2 = self.weight_variable([5, 5, 32, 64])
          self.b_conv2 = self.bias_variable([64])
          # relu
          self.h_conv2 = tf.nn.relu(self.conv2d(self.h_pool1, self.W_conv2) + self.b_conv2)
          #max pool
          self.h_pool2 = self.max_pool_2x2(self.h_conv2) 
          # fully connected layer
          self.W_fc1 = self.weight_variable([7 * 7 * 64, 1024])
          self.b_fc1 = self.bias_variable([1024])
          self.h_pool2_flat = tf.reshape(self.h_pool2, [-1, 7*7*64])
          #relul
          self.h_fc1 = tf.nn.relu(tf.matmul(self.h_pool2_flat, self.W_fc1) + self.b_fc1) 
          #dropout layer build
          self.keep_prob = tf.placeholder(tf.float32)
          self.h_fc1_drop = tf.nn.dropout(self.h_fc1, self.keep_prob)
          #output layer L
          self.W_fc2 = self.weight_variable([1024, 1])
          self.b_fc2 = self.bias_variable([1])
          self.y_conv = tf.matmul(self.h_fc1_drop, self.W_fc2) + self.b_fc2
          # The raw formulation of cross-entropy,
          #
          #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
          #                                 reduction_indices=[1]))
          #
          # can be numerically unstable.
          #
          # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
          # outputs of 'y', and then average across the batch.
          
          #define loss function and training step - Adam vs GradientDescent
          self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.y_conv))
          self.mse_loss = tf.reduce_mean(tf.squared_difference(self.y_, self.y_conv))
          self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.mse_loss)
          #train_step = tf.train.GradientDescentOptimizer(1e-2).minimize(cross_entropy)
          #define correct and accuracy of the model for testing
          self.correct_prediction = tf.equal(tf.argmax(self.y_conv,1), tf.argmax(self.y_,1))
          self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
          
  def init_test_data(self, params):
      # Train
      #ipdb.set_trace()
      self.testdata = self.testdata.sort_index()
      self.xtest = self.testdata[self.testdata.columns[:-1]] 
      self.ytest = self.testdata[self.testdata.columns[-1]] 
      # setup tf session


  def weight_variable(self, shape):
      '''input shape of weight in the form [input, output]
         output Variable tf holder
      '''
      initial = tf.truncated_normal(shape, stddev=0.1)
      return tf.Variable(initial)

  def bias_variable(self, shape):
      '''input shape of bias in the form [input, output]
         output Variable tf holder
      '''
      initial = tf.constant(0.1, shape=shape)
      return tf.Variable(initial)

  def nn_train(self, params):
      with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
          #sess = tf.InteractiveSession()
          #init = tf.initialize_all_variables()
          #sess.run(init)
          sess.run(tf.global_variables_initializer())
          self.start_time = time.time()
          self.mini_batch_size = 200
          self.iters = 1000
          #batch = tf.train.batch([xtrain, ytrain], batch_size=100)
          for i in range(self.iters):
              self.traindata = self.traindata.reindex(np.random.permutation(self.traindata.index))
              self.batches = [ self.traindata[k:k+self.mini_batch_size] for k in xrange(0,len(self.traindata), self.mini_batch_size)]
              for batch in self.batches:
                  if len(batch) == self.mini_batch_size:
                    batchx = batch[batch.columns[:-1]]
                    batchy = batch[batch.columns[-1]]
                        
                    yvals = np.array(batchy).reshape((len(batchy),1))
                    #print(yvals.shape, iteration) 
                    self.train_step.run(feed_dict={self.x: batchx, self.y_: yvals, self.keep_prob: 0.5})
              #monitoring condition
              if i%100 == 0:
                print('time elapsed at iteration %s: %s'%(i, time.time() - self.start_time))
                #train_accuracy = accuracy.eval(feed_dict={x:batchx, y_: yvals, keep_prob: 1.0})
                #print("step %d, training accuracy %g"%(i, train_accuracy))
                #test_accuracy = accuracy.eval(feed_dict={x:xtest, y_: ytvals, keep_prob: 1.0})
                #print("step %d, test accuracy %g"%(i, test_accuracy))
          #Test
          #print("final test accuracy %g"%accuracy.eval(feed_dict={x: xtest, y_: ytvals, keep_prob: 1.0}))  
          feed_dict={self.x: self.xtest, self.keep_prob:1.0}
          classification = self.y_conv.eval(feed_dict)
          print('size: %s %s'%(len(classification), self.ytest.shape))
          sum1 = 0.0
          sum2 = 0.0
          for c in zip(classification, self.ytest):
              print(c, 'delta: %s'%(c[1]-c[0]))
              sum1 += c[1] - c[0]
              sum2 += (c[1] - c[0])**2
          print(sum1, sum1/float((len(self.ytest))), sum2/float(len(self.ytest)))
          print('***************************finished******************************')

  def conv2d(self,x, W):
      return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

  def max_pool_2x2(self,x):
      return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME')

  def vectorized(self,j):
      e = np.zeros((2,1))
      e[int(j)] = 1.0
      return e


def input_fn(data_set, x_vals, y_vals):
  '''input dataframe, x cols and y col
     output tensor for x and y
  '''
  feature_cols = {k: tf.constant(data_set[k].values)
                  for k in x_vals}
  labels = tf.constant(data_set[y_vals].values)
  return feature_cols, labels





if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  #tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
  fn = FinNet(FLAGS)
