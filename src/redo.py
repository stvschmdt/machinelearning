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

FLAGS=None


def weight_variable(shape):
    initial = tf.trucated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1],strides=[1,2,2,1], padding='SAME')

def main():
    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])
    #layer 1
    x_image = tf.reshape(x, [-1,28,28,1])
    W_conv1 = weight_variable([5,5,1,32])
    b_conv1 = bias_variable([32])
    # 784 -> 28x28 -> 14x14
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    #layer 2
    W_conv2 = weight_variable([5,5, 32, 64])
    b_conv2 = bias_variable([64])
    # 784 -> 14x14 -> 7xy
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    #fully connected layer
    W_fc1 = weight_variable([7*7*64,1024])
    b_fc1 = bias_variable([1024])

    #final layer
    h_pool2_flat = tf.reshape([-1,7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    #drop out before final layer
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_dropout = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024,10])
    b_fc2 = bias_variable([10])

    y_conv = tf.matmul(h_fc1_dropout, W_fc2) + b_fc2

    #get some data
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    #init a session
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())




if __name__ == '__main__':
    main()
