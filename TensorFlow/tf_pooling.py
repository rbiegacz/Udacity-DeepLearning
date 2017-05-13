#
# Copyright (c) 2017 Rafal Biegacz
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
"""
    this silly script shows how to use Tensorflow max_pool and avg_pool instructions.
"""

#pylint: disable=C0325
#pylint: disable=E1103
#pylint: disable=F0401

import os
import tensorflow as tf
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def tf_pooling_example():
    """
        this silly script shows how to use Tensorflow max_pool and avg_pool instructions.
        for reference to max_pool, avg_pool refer to:
        https://www.tensorflow.org/versions/r0.10/api_docs/python/nn/pooling
    """
    input_data = np.array([\
            [0, 1, 0.5, 10],\
            [2, 2.5, 1, -8],\
            [4, 0, 5, 6],\
            [15, 1, 2, 3]],\
            dtype=np.float32).reshape((1, 4, 4, 1))

    input_layer = tf.placeholder(tf.float32, (1, 4, 4, 1))
    #Set the ksize (filter size) for each dimension (batch_size, height, width, depth)
    filter_shape = [1, 2, 2, 1]
    #Strides are defined like that(batch_size, height, width, depth)
    strides = [1, 2, 2, 1]
    padding = 'VALID'
    max_pool = tf.nn.max_pool(input_layer, filter_shape, strides, padding)
    avg_pool = tf.nn.avg_pool(input_layer, filter_shape, strides, padding)

    print("Input data...")
    print(input_data)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print("\nThis is an example of MAX Pooling...")
        print(sess.run(max_pool, feed_dict={input_layer: input_data}))
        print("\nThis is an example of Average Pooling...")
        print(sess.run(avg_pool, feed_dict={input_layer: input_data}))

tf_pooling_example()
