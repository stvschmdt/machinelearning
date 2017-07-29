# ==============================================================================

"""an adversarial attack against a black box machine learning or artificial
   intelligence entity. train test against an oracle -> pertrurb input vector
   attain transferability to misclassification in original oracle model
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import numpy as np
import tensorflow as tf
from blackbox import BlackBox
from logger import Logging

FLAGS = None
logger = Logging()
# ==============================================================================


def deepnn(x):
  """deepnn builds the graph for a deep net for classifying digits.
  Args:
    x: an input tensor with the dimensions (N_examples, 784) for example
  Returns:
    A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
    equal to the logits of classifying the digit into one of 10 classes (the
    digits 0-9). keep_prob is a scalar placeholder for the probability of
    dropout
  """
  # Reshape to use within a convolutional neural net.
  # Last dimension is for "features" - there is only one here, since images are
  # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
  x_image = tf.reshape(x, [-1, 28, 28, 1])

  # First convolutional layer - maps one grayscale image to 32 feature maps.
  W_conv1 = weight_variable([5, 5, 1, 32])
  b_conv1 = bias_variable([32])
  h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

  # Pooling layer - downsamples by 2X.
  h_pool1 = max_pool_2x2(h_conv1)

  # Second convolutional layer -- maps 32 feature maps to 64.
  W_conv2 = weight_variable([5, 5, 32, 64])
  b_conv2 = bias_variable([64])
  h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

  # Second pooling layer.
  h_pool2 = max_pool_2x2(h_conv2)

  # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
  # is down to 7x7x64 feature maps -- maps this to 1024 features.
  W_fc1 = weight_variable([7 * 7 * 64, 1024])
  b_fc1 = bias_variable([1024])

  h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
  h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  # Dropout - controls the complexity of the model, prevents co-adaptation of
  # features.
  keep_prob = tf.placeholder(tf.float32)
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  # Map the 1024 features to 10 classes, one for each digit
  W_fc2 = weight_variable([1024, 10])
  b_fc2 = bias_variable([10])

  y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
  return y_conv, keep_prob


def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def one_hot(x):
    y = np.zeros((10,1))
    y[x] = 1.0
    return y

#goodfellow et al attack x'=x+epsilon*sgn(gradient)
def goodfellow_mod(x, grad, epsilon=0.05):
    xprime =  x + np.sign(grad)*epsilon
    #dumb way to keep within 0,1, empirical testing will determine keeping
    xprime[ xprime < 0.0] = 0
    xprime[ xprime > 1.0] = 0
    return xprime

#convert a list of tuples into nparray
def image_list_to_np(l_image, idx):
    vals = [ x[idx] for x in l_image ]
    vals = np.array(vals)
    return vals

#split data arrays with images and labels into train/test set
def split_train_data(xarr, yarr, n):
  train_images = np.array(xarr[:-n])
  train_labels = np.array(yarr[:-n])
  test_images =  np.array(xarr[-n:])
  test_labels =  np.array(yarr[-n:])
  return train_images, train_labels, test_images, test_labels

def main(_):
  # import data
  mdl = BlackBox(FLAGS)
  # NOTE this will work with format: mdl.oracle = (image, pred_val, true_val)
  logger.info('obtained black box training data')
  mnist = mdl.oracle
  # translate into tensorflow style nparrays
  x_vals = image_list_to_np(mnist, 0)
  true_vals = image_list_to_np(mnist,2)
  
  # yvals converted to one hot vector
  y_vals = [ x[1] for x in mnist ]
  y_vals = [ one_hot(i) for i in y_vals]
  y_vals = np.array(y_vals)
  y_vals = y_vals.reshape((len(y_vals),10))
 
  # Tensorflow variable setup

  # input vector
  x = tf.placeholder(tf.float32, [None, 784])
  # y output vector
  y_ = tf.placeholder(tf.float32, [None, 10])
  # build the graph for the deep net
  y_conv, keep_prob = deepnn(x)
  # define loss function -> cross entropy for now with softmax
  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
  # train step, 1e-4 is default, best to use -2/-3 depending on time
  train_step = tf.train.AdamOptimizer(FLAGS.optimize).minimize(cross_entropy)
  # define correct prediction vectore and accuracy comparison
  correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
 
  # split training and test data into nparrays
  train_images, train_labels, test_images, test_labels =split_train_data(x_vals, y_vals, FLAGS.split)
  logger.info('training sets: %d test sets: %d' % (len(train_images),len(test_images)))
  cnn_saver = tf.train.Saver()
  logger.info('starting adversarvial model training')
  #begin training with session
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # TODO create batching loop
    for i in range(FLAGS.iters):
      #sanity check on accuracy should be going down -> necessary but not sufficient
      if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: train_images, y_: train_labels, keep_prob: 1.0})
        logger.info('step %d, training accuracy %g' % (i, train_accuracy))
      #update rule - softmax vector obtained for checking
      trainer, softmax = sess.run([train_step, cross_entropy],feed_dict={x: train_images, y_: train_labels, keep_prob: 0.5})
    logger.info('adversarial model has been trained')
    # snag the gradient vector wrt X inputs
    grads = tf.gradients(cross_entropy, [x])
    jacobian = sess.run(grads, feed_dict={x:test_images, y_:test_labels, keep_prob: 1.0})
    #use test data as input for perturbations
    test_ = tf.argmax(y_,1)
    test_vals = test_.eval(feed_dict={y_:test_labels})
    pred_ = tf.argmax(y_conv,1)
    pred_vals = pred_.eval(feed_dict={x:test_images, y_:test_labels, keep_prob:1.0})
    true_pred = [ (pxl, p) for pxl, p, r in zip(test_images, pred_vals, test_vals) if p==r ]
    logger.info('true positive test exemplars: %s'.format(len(true_pred)))
    logger.results('adversary accuracy: %g' % (accuracy.eval(feed_dict={x: test_images, y_: test_labels, keep_prob: 1.0})))
    #setup the goodfellow attack iterations
    adv_list = []
    for idx,pos in enumerate(true_pred):
      for epsilon in np.linspace(0.025,.25,num=FLAGS.augments):
          #xprime = []
          #yprime = []
          xp = goodfellow_mod(np.array(pos[0]), jacobian[0][idx], epsilon)
          prime_label = one_hot(int(pos[1]))
          #xprime.append(xp)
          #yprime.append(prime_label)
          xprime = np.array(xp)
          xprime = xprime.reshape((1,784))
          yprime = np.array(prime_label)
          yprime = yprime.reshape((1, 10))
          acc = accuracy.eval(feed_dict={x: xprime, y_: yprime, keep_prob: 1.0})
          #corr = sess.run(mdl.y, feed_dict={x:xprime})
          if acc < 0.5:
              adv_list.append((xprime, np.argmax(yprime), np.argmax(test_vals[idx]), epsilon))
              print('YES!!!',np.argmax(yprime), test_vals[idx], epsilon)
              #logger.results('YES adversary accuracy: %g %f' % (acc, epsilon))
              break
          logger.results('NO adversary accuracy: %g %f' % (acc, epsilon))
        #can do this each iteration - or as a whole...at this point timing doesnt matter, but will

    # save model to file
    cnn_saver_path = cnn_saver.save(sess, 'cnn_saver.ckpt')
    
    # at this point adv_list is a tuple (x modifed image, y label, true label, epsilon found) 
    adv_images = [ a[0] for a in adv_list ]
    l = len(adv_list)
    adv_images = np.array(adv_images)
    import ipdb
    #ipdb.set_trace()
    adv_images = adv_images.reshape((l, 784))
    adv_labels = [ a[1] for a in adv_list ]
    adv_labels = [ one_hot(int(v)) for v in adv_labels ]
    adv_labels = np.array(adv_labels)
    adv_labels = adv_labels.reshape((l,10))
    adv_epsilon = [ a[2] for a in adv_list ]
    adv_epsilon = np.array(adv_epsilon)
    # test for transferability
    #mdl.correct_prediction = tf.equal(tf.argmax(mdl.y, 1), tf.argmax(mdl.y_, 1))
    #mdl.accuracy = tf.reduce_mean(tf.cast(mdl.correct_prediction, tf.float32))
    #transfer = mdl.correct_prediction.eval(feed_dict={mdl.x: adv_images, mdl.y_: adv_labels})
    adv_pred = sess.run(tf.argmax(adv_labels,1))
    adv_ = tf.argmax(true_pred,1)
    adv_real = sess.run(adv_, feed_dict={mdl.x: adv_images})
    for a in zip(adv_images, adv_preds, true_pred):
        print(a)
    logger.info('****************** simple model accuracy **************')
    logger.results('black box adversarial attack transferability: ' % (sess.run(mdl.accuracy, feed_dict={mdl.x: adv_images,mdl.y_: adv_labels})))


if __name__ == '__main__':
# args: batch_size, graddescent number, training iters, epsilon trial space
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str,default='/tmp/tensorflow/mnist/input_data',help='directory for storing input data')
  parser.add_argument('--batch_size', type=int,default=100,help='batch size for stochastic gradient descent')
  parser.add_argument('--optimize', type=float,default=1e-3,help='threshold for adam optimizer')
  parser.add_argument('--iters', type=int,default=20,help='cnn training epochs')
  parser.add_argument('--augments', type=int,default=10,help='number of attack augmentations to sample for epsilon on inputs')
  parser.add_argument('--split', type=float,default=200,help='train test set percent split')
  parser.add_argument('--fsplit', type=float,default=None,help='train test set percent split')
  FLAGS, unparsed = parser.parse_known_args()
  #logger.info('FLAGS set: %g'%(FLAGS))
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)






