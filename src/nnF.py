# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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

def input_fn(data_set, x_vals, y_vals):
  '''input dataframe, x cols and y col
     output tensor for x and y
  '''
  feature_cols = {k: tf.constant(data_set[k].values)
                  for k in x_vals}
  labels = tf.constant(data_set[y_vals].values)
  return feature_cols, labels

def weight_variable(shape):
  '''input shape of weight in the form [input, output]
     output Variable tf holder
  '''
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  '''input shape of bias in the form [input, output]
     output Variable tf holder
  '''
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def vectorized(j):
  e = np.zeros((2,1))
  e[int(j)] = 1.0
  return e

def main(_):
  # Import data
  start_time = time.time()
  rdr = Reader()
  rdr.read_images('/home/ubuntu/store/fin_data', 'csv',True, False)
  xvals = np.array(rdr.c_images)
  yholder = {}
  with open('/home/ubuntu/store/fin_data/99999.y_vals.csv', 'r') as f:
    r = csv.reader(f)
    for row in r:
      yholder[int(row[0])] = float(row[1])
  yvals = [yholder[int(x)] for x in rdr.file_labels]
  yvals = np.array(yvals)
  xvals = pd.DataFrame(xvals)
  xvals['y'] = yvals
  #traindata, testdata = train_test_split(xvals, test_size=.25)
  traindata = xvals.loc[:1500]
  testdata = xvals.loc[1500:]
  print('read time: %s train size: %s test size: %s'%((time.time()-start_time),len(traindata),len(testdata)))
  x = tf.placeholder(tf.float32, [None, 784])
  W = tf.Variable(tf.zeros([784, 1]))
  b = tf.Variable(tf.zeros([1]))
  y = tf.matmul(x, W) + b

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 1])

  # create the convolutional hidden layer sizes - make even more generic in future
  W_conv1 = weight_variable([5, 5, 1, 32])
  b_conv1 = bias_variable([32])
  x_image = tf.reshape(x, [-1,28,28,1])
  # relu
  h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
  # max pool
  h_pool1 = max_pool_2x2(h_conv1)
  # create second convolution layer
  W_conv2 = weight_variable([5, 5, 32, 64])
  b_conv2 = bias_variable([64])
  # relu
  h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
  #max pool
  h_pool2 = max_pool_2x2(h_conv2) 
  # fully connected layer
  W_fc1 = weight_variable([7 * 7 * 64, 1024])
  b_fc1 = bias_variable([1024])
  h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
  #relul
  h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1) 
  #dropout layer build
  keep_prob = tf.placeholder(tf.float32)
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
  #output layer L
  W_fc2 = weight_variable([1024, 1])
  b_fc2 = bias_variable([1])
  y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


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
  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
  mse_loss = tf.reduce_mean(tf.squared_difference(y_, y_conv))
  train_step = tf.train.AdamOptimizer(1e-4).minimize(mse_loss)
  #train_step = tf.train.GradientDescentOptimizer(1e-2).minimize(cross_entropy)
  #define correct and accuracy of the model for testing
  correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  

  # Train
  #ipdb.set_trace()
  testdata = testdata.sort_index()
  xtest = testdata[testdata.columns[:-1]] 
  ytest = testdata[testdata.columns[-1]] 
  # setup tf session
  sess = tf.InteractiveSession()
  #init = tf.initialize_all_variables()
  #sess.run(init)
  sess.run(tf.global_variables_initializer())
  start_time = time.time()
  #batch = tf.train.batch([xtrain, ytrain], batch_size=100)
  for i in range(3000):
      mini_batch_size = 50
      traindata = traindata.reindex(np.random.permutation(traindata.index))
      batches = [ traindata[k:k+mini_batch_size] for k in xrange(0,len(traindata), mini_batch_size)]
      for batch in batches:
          if len(batch) == 50:
            batchx = batch[batch.columns[:-1]]
            batchy = batch[batch.columns[-1]]
                
            yvals = np.array(batchy).reshape((len(batchy),1))
            #print(yvals.shape, iteration) 
            train_step.run(feed_dict={x: batchx, y_: yvals, keep_prob: 0.5})
      #monitoring condition
      if i%100 == 0:
        print('time elapsed at iteration %s: %s'%(i, time.time() - start_time))
        #train_accuracy = accuracy.eval(feed_dict={x:batchx, y_: yvals, keep_prob: 1.0})
        #print("step %d, training accuracy %g"%(i, train_accuracy))
        #test_accuracy = accuracy.eval(feed_dict={x:xtest, y_: ytvals, keep_prob: 1.0})
        #print("step %d, test accuracy %g"%(i, test_accuracy))
  #Test
  #print("final test accuracy %g"%accuracy.eval(feed_dict={x: xtest, y_: ytvals, keep_prob: 1.0}))  
  feed_dict={x: xtest, keep_prob:1.0}
  classification = y_conv.eval(feed_dict)
  print('size: %s %s'%(len(classification), ytest.shape))
  sum1 = 0.0
  sum2 = 0.0
  for c in zip(classification, ytest):
      print(c, 'delta: %s'%(c[1]-c[0]))
      sum1 += c[1] - c[0]
      sum2 += (c[1] - c[0])**2
  print(sum1, sum1/float((len(ytest))), sum2/float(len(ytest)))
  
  #print("real correct: %s"%(correct/float(len(classification))))
  #cp = correct_prediction.eval(feed_dict={x:xtest, y_: ytvals, keep_prob: 1.0}, session=sess)
  #print('argmax version:')
  #cp1 = tf.argmax(y_conv,1)
  #cp2 = cp1.eval(feed_dict)
  #correct = 0
  #for c in zip(cp2, ytvals):
  #    if c[0] == c[1].argmax():
  #        correct += 1
  #        #print(c[0], c[1].argmax(), 'True')
  #    else:
  #        correct += 0
          #print(c[0], c[1].argmax(), 'False')
  #print("real correct: %s"%(correct/float(len(cp2))))
  #print("test check \n%s"%zip(cp, ytvals))  
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
