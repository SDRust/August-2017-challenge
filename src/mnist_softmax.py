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
https://www.tensorflow.org/get_started/mnist/beginners
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.saved_model.builder import SavedModelBuilder
from tensorflow.python.saved_model.signature_def_utils import build_signature_def
from tensorflow.python.saved_model.signature_constants import REGRESS_METHOD_NAME
from tensorflow.python.saved_model.tag_constants import TRAINING, SERVING
from tensorflow.python.saved_model.utils import build_tensor_info

import tensorflow as tf

FLAGS = None


def main(_):
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  # Create the model
  x = tf.placeholder(tf.float32, [None, 784])
  W = tf.Variable(tf.zeros([784, 10]))
  b = tf.Variable(tf.zeros([10]))
  y = tf.matmul(x, W) + b
  prediction = tf.softmax(y)

  # Ground truth labels
  y_ = tf.placeholder(tf.float32, [None, 10])

  # Define loss and optimizer
  # The raw formulation of cross-entropy,
  #
  #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
  #                                 reduction_indices=[1]))
  #
  # can be numerically unstable.
  #
  # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
  # outputs of 'y', and then average across the batch.
  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy,
          name='train')

  sess = tf.InteractiveSession()

  builder = SavedModelBuilder('saved_model')

  tf.global_variables_initializer().run()
  # Train
  for _ in range(2000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

  # Test trained model
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                      y_: mnist.test.labels}))

  signature_inputs = {
      "x": build_tensor_info(x),
      "y": build_tensor_info(y_)
  }
  signature_outputs = {
      "out": build_tensor_info(prediction)
  }
  signature_def = build_signature_def(
      signature_inputs, signature_outputs,
      REGRESS_METHOD_NAME)
  builder.add_meta_graph_and_variables(
      sess, [TRAINING, SERVING],
      signature_def_map={
          REGRESS_METHOD_NAME: signature_def
      },
      assets_collection=tf.get_collection(tf.GraphKeys.ASSET_FILEPATHS))
  builder.save(as_text=False)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
