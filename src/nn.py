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

"""A very simple nn classifier.
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
import ipdb

def main(_):
  # Import data
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
  #cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

  #cross_entropy = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(y,1e-10,1.0)))
  cross_entropy = tf.reduce_sum(tf.square(tf.subtract(y_,y)))
  train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()
  # Train
  y_test = y_test.as_matrix()
  for i in range(20):
    #batch_xs, batch_ys = mnist.train.next_batch(100)
    #batch_xs = X_train
    #batch_ys = y_train.as_matrix()
      sess.run(train_step, feed_dict={x: X_train, y_: y_train.as_matrix()})
  # Test trained model
      print("[model] training is complete ***************** ")
      correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
      accuracy = tf.reduce_mean(tf.subtract(tf.cast(correct_prediction, tf.float32), y_test[:10000]))
      
      print('accuracy: %s'%sess.run(accuracy, feed_dict={x: X_test.head(10000),
          y_: y_test[:10000]}))
  #cp = sess.run(tf.cast(correct_prediction, tf.float32), feed_dict={x: X_test.head(10000), y_: y_test[:10000]})
  #lacc = tf.subtract(tf.cast(correct_prediction, tf.float32), y_test[:10000])
  #cp = sess.run(lacc, feed_dict={x: X_test.head(10000), y_ : y_test[:10000]})
  #count = 0
  #for idx, c in enumerate(cp):
      #if c != y_test[idx]:
          ##print(idx, c, y_test[idx])
          #continue
      #else:
          #count +=1
  #print((count/float(10000)))
  sess.close()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
