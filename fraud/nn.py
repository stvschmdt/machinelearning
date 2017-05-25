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

"""A very simple MNIST classifier.
See extensive documentation at
http://tensorflow.org/tutorials/mnist/beginners/index.md
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import tensorflow as tf

from processing import Processor
from logger import Logging
FLAGS = None


def main(_):
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  CSV_FILE = '~/store/fraud_data/creditcard.csv'
  YCOL = 'Class'
  logger = Logging()
  proc = Processor()


#TODO make this test suite
  data = proc.load_csv(CSV_FILE)
  data = proc.normalize_col(data, 'Amount')
  data = data.drop(['Time'],axis=1)
  X = proc.get_xvals(data, YCOL)
  y = proc.get_yvals(data, YCOL)
#print data.describe()
  Xu, yu = proc.under_sample(data, YCOL)
  Xu_train, Xu_test, yu_train, yu_test = proc.cross_validation_sets(Xu, yu,.3,0)
  X_train, X_test, y_train, y_test = proc.cross_validation_sets(X, y,.3,0)
  x = tf.placeholder(tf.float32, [None, 29])
  W = tf.Variable(tf.zeros([29, 1]))
  b = tf.Variable(tf.zeros([1]))
  y = tf.matmul(x, W) + b

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 1])

  # The raw formulation of cross-entropy,
  #
  #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
  #                             )    reduction_indices=[1]))
  #
  # can be numerically unstable.
  #
  # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
  # outputs of 'y', and then average across the batch.
  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()
  # Train
  for i in range(1):
    #batch_xs, batch_ys = mnist.train.next_batch(100)
    batch_xs, batch_ys = X_train, y_train.iloc[:,0:].values

    #print(mnist.test.images.shape, mnist.test.labels.shape, mnist.train.labels.shape, batch_ys.shape)
    #if True:
        #return
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
  # Test trained model
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  print('accuracy: %s'%sess.run(accuracy, feed_dict={x: X_test,
      y_: y_test.iloc[:,0:].values}))
  
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
