#
# Copyright (c) 2016 Rafal Biegacz
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

""" Simple Neural Network with one hidden layer """

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
import sys

output = None

if len(sys.argv) > 1:
    print("Random weights initialization...")
    hidden_layer_weights = tf.Variable(tf.random_normal([4, 3]))
    out_weights = tf.Variable(tf.random_normal([3, 2]))
else:
    print("Predefined values used as weights...")
    hidden_layer_weights = [[0.1, 0.2, 0.4],[0.4, 0.6, 0.6],[0.5, 0.9, 0.1], [0.8, 0.2, 0.8]]
    out_weights = [[0.1, 0.6], [0.2, 0.1], [0.7, 0.9]]

print("hidden_layer_weights:")
print(format(hidden_layer_weights))
print("out_weights:")
print(format(out_weights))

# Weights and biases
weights = [hidden_layer_weights, out_weights]
biases = [tf.Variable(tf.zeros(3)), tf.Variable(tf.zeros(2))]
# Input
features = tf.Variable([[1.0, 2.0, 3.0, 4.0], [-1.0, -2.0, -3.0, -4.0], [11.0, 12.0, 13.0, 14.0]])


# Model of the Neural Network - one hidden layer with RELU as activation function
hidden_layer = tf.add(tf.matmul(features, weights[0]), biases[0])
hidden_layer = tf.nn.relu(hidden_layer)
output = tf.add(tf.matmul(hidden_layer, weights[1]), biases[1])

# Running TensorFlow session
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    print(sess.run(output))
