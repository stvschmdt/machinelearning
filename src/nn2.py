# ==============================================================================
# Author: Steve Schmidt
# 2-D NN using READER and LOADER for Specific Data Set (Images)
# Extended from TensorFlow Documentation
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
  rdr = Reader()
  rdr.read_images('../../store/images_data/train/', 'dog', True, True)
  xdf = pd.DataFrame(rdr.d_images)
  x2df = pd.DataFrame(rdr.c_images)
  # xdf and x2df are both dataframes for dogs/cats separately -> could be used generically for 2 class reads
  xdf.columns = [ 'x%s'%x for x in range(xdf.shape[1]) ]
  x2df.columns = [ 'x%s'%x for x in range(x2df.shape[1]) ]
  #dogs 1, cats zero
  xdf['y'] = 1
  x2df['y'] = 0

  #cleanup and concat dataframes - delete null rows upfront
  xdf = pd.concat([xdf, x2df], ignore_index=True)
  xdf = xdf.dropna(how='any')
  xdf = xdf.reindex(np.random.permutation(xdf.index))
  traindata, testdata = train_test_split(xdf, test_size=.2)
  xcol = xdf.columns[:-1]
  ycol = xdf.columns[-1]
  traindata = traindata.dropna()
  traindata = traindata[~traindata.isnull()]
  testdata = testdata.dropna()
  testdata = traindata[~traindata.isnull()]
  
  
  # create initial x, W, b, y variables and sizes (canocal form)
  x = tf.placeholder(tf.float32, [None, 784])
  W = tf.Variable(tf.zeros([784, 2]))
  b = tf.Variable(tf.zeros([2]))
  y = tf.matmul(x, W) + b

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 2])

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
  W_fc2 = weight_variable([1024, 2])
  b_fc2 = bias_variable([2])
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
  train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
  #train_step = tf.train.GradientDescentOptimizer(1e-2).minimize(cross_entropy)
  #define correct and accuracy of the model for testing
  correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  

  # format the test data into tf tensor readable np arrays
  testdata = testdata[~testdata.isnull().any(axis=1)]
  xtest = testdata[testdata.columns[:-1]]
  ytest = testdata[testdata.columns[-1]]
  ytvals = [vectorized(j) for j in ytest.values]
  ytvals = np.array(ytvals)
  ytvals = ytvals.reshape((len(ytvals), 2))
  # Train
  #ipdb.set_trace()
  
  # setup tf session
  sess = tf.InteractiveSession()
  #init = tf.initialize_all_variables()
  #sess.run(init)
  sess.run(tf.global_variables_initializer())
  #batch = tf.train.batch([xtrain, ytrain], batch_size=100)
  for i in range(1000):
      mini_batch_size = 100
      traindata = traindata.reindex(np.random.permutation(xdf.index))
      traindata = traindata[~traindata.isnull().any(axis=1)]
      batches = [ traindata[k:k+mini_batch_size] for k in xrange(0,len(traindata), mini_batch_size)]
      for batch in batches:
          if len(batch) == 100:
            batchx = batch[batch.columns[:-1]]
            batchy = batch[batch.columns[-1]]
                
            yvals = [ vectorized(j) for j in batchy.values ]
            yvals = np.array(yvals)
            yvals = yvals.reshape((len(yvals),2))
            #print(yvals.shape, iteration) 
            train_step.run(feed_dict={x: batchx, y_: yvals, keep_prob: 0.5})
      #monitoring condition
      if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x:batchx, y_: yvals, keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
        #test_accuracy = accuracy.eval(feed_dict={x:xtest, y_: ytvals, keep_prob: 1.0})
        #print("step %d, test accuracy %g"%(i, test_accuracy))
  #Test
  print("final test accuracy %g"%accuracy.eval(feed_dict={x: xtest, y_: ytvals, keep_prob: 1.0}))  
  feed_dict={x: xtest, keep_prob:1.0}
  classification = y_conv.eval(feed_dict)
  print('size: %s %s'%(len(classification), ytest.shape))
  correct = 0
  for c in zip(classification, ytvals):
      if c[0].argmax() == c[1].argmax():
          correct += 1
          #print(c, c[0].argmax(), c[1].argmax(), 'True')
      else:
          correct += 0
          #print(c, c[0].argmax(), c[1].argmax(), 'False')
  print("real correct: %s"%(correct/float(len(classification))))
  cp = correct_prediction.eval(feed_dict={x:xtest, y_: ytvals, keep_prob: 1.0}, session=sess)
  #print('argmax version:')
  cp1 = tf.argmax(y_conv,1)
  cp2 = cp1.eval(feed_dict)
  correct = 0
  for c in zip(cp2, ytvals):
      if c[0] == c[1].argmax():
          correct += 1
          #print(c[0], c[1].argmax(), 'True')
      else:
          correct += 0
          #print(c[0], c[1].argmax(), 'False')
  print("real correct: %s"%(correct/float(len(cp2))))
  #print("test check \n%s"%zip(cp, ytvals))  
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
