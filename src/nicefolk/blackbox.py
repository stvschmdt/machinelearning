# ==============================================================================

"""a simple DNN classifier model class.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

import numpy as np
import tensorflow as tf
from logger import Logging

FLAGS = None
logger = Logging()

# ==============================================================================


class BlackBox(object):

  def __init__(self, FLAGS, filename='simple_nn.ckpt'):
      # Import data
      self.mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
      # Create the model
      self.x = tf.placeholder(tf.float32, [None, 784])
      self.W = tf.Variable(tf.zeros([784, 10]))
      self.b = tf.Variable(tf.zeros([10]))
      self.y = tf.matmul(self.x, self.W) + self.b

      # Define loss and optimizer
      self.y_ = tf.placeholder(tf.float32, [None, 10])

      # The raw formulation of cross-entropy,
      #
      #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
      #                                 reduction_indices=[1]))
      #
      # can be numerically unstable.
      #
      # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
      # outputs of 'y', and then average across the batch.
      cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.y))
      train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
      simple_saver = tf.train.Saver()
      self.sess = tf.InteractiveSession()
      tf.global_variables_initializer().run()
      # Train
      logger.info('simple nn model accuracy training...')
      for _ in range(1000):
        self.batch_xs, self.batch_ys = self.mnist.train.next_batch(100)
        self.sess.run(train_step, feed_dict={self.x: self.batch_xs, self.y_: self.batch_ys})

      # Test trained model
      self.correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
      self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
      self.reals = self.sess.run(tf.argmax(self.mnist.test.labels,1))
      self.preds = tf.argmax(self.y,1)
      self.preds = self.sess.run(self.preds, feed_dict={self.x: self.mnist.test.images})
      self.oracle = []
      for c in zip(self.mnist.test.images, self.preds, self.reals):
          self.oracle.append(c)
      logger.results('black box accuracy: %g' % (self.sess.run(self.accuracy, feed_dict={self.x: self.mnist.test.images,self.y_: self.mnist.test.labels})))
      simple_saver_path = simple_saver.save(self.sess, 'simple.ckpt')
      self.true_positives = [ ex for ex in self.oracle if ex[1] == ex[2] ]
      self.pictruelabel = (self.true_positives[0][1], self.true_positives[0][2])
      self.pictrue = self.true_positives[0][0].reshape((28,28))
      self.true_negatives = [ ex for ex in self.oracle if ex[1] != ex[2] ]
      self.picfalselabel = (self.true_negatives[0][1], self.true_negatives[0][2])
      self.picfalse = self.true_negatives[0][0].reshape((28,28))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,default='/tmp/tensorflow/mnist/input_data',help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    bb = BlackBox(FLAGS)
    #print(bb.oracle[0], bb.true_positives[0])
    #print(float(len(bb.true_positives)/float(len(bb.oracle))))
