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

#pylint: disable=C0325
#pylint: disable=E1103
#pylint: disable=F0401
"""
    simple script based on TF Learn that trains the model for MNIST Digits recognition
    the script tries to discover the optimal network for digits recognition
"""

import os
import numpy as np
import tensorflow as tf
import tflearn
import tflearn.datasets.mnist as mnist

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def load_data():
    """
        this function loads MNIST data (downloads it if necessary)
    """
    trainX, trainY, test_x, testY = mnist.load_data(one_hot=True)
    return trainX, trainY, test_x, testY

def build_model(nunits_hidden1=128, nunits_hidden2=28, learning_rate=0.1):
    """
        This function builds model for digits recognition
    """
    # Include the input layer, hidden layer(s), and set how you want to train the model
    # Input layer
    net = tflearn.input_data([None, 784])
    # Hidden Layer #1
    net = tflearn.fully_connected(net, nunits_hidden1, activation='ReLU')
    # Hidden Layer #2
    net = tflearn.fully_connected(net, nunits_hidden2, activation='ReLU')
    # Output Layer #1
    net = tflearn.fully_connected(net, 10, activation='softmax')
    net = tflearn.regression(net, optimizer='sgd',\
                            learning_rate=learning_rate, loss='categorical_crossentropy')
    model = tflearn.DNN(net)
    return model

def train_model(model, trainX, trainY, batch_size=50, n_epoch=20):
    """
        this function trains the model
    """
    model.fit(trainX, trainY, validation_set=0.1,\
              show_metric=True, batch_size=batch_size, n_epoch=n_epoch)
    return model

def test_model(model, text_x, text_y):
    """
        this function tests the model
    """
    predictions = np.array(model.predict(text_x)).argmax(axis=1)
    actual = text_y.argmax(axis=1)
    test_accuracy = np.mean(predictions == actual, axis=0)
    print("Test accuracy: ", test_accuracy)
    return test_accuracy

if __name__ == "__main__":
    batch_size = 50
    n_epochs = [10, 20, 50, 100]
    nunits_hidden1s = [28, 56, 128, 256, 512]
    nunits_hidden2s = [28, 56, 128, 256, 512]
    results = {}

    trainX, trainY, test_x, testY = load_data()
    for n_epoch in n_epochs:
        for nunits_hidden1 in nunits_hidden1s:
            for nunits_hidden2 in nunits_hidden2s:
                tf.reset_default_graph()
                model = build_model(nunits_hidden1=nunits_hidden1, nunits_hidden2=nunits_hidden2)
                old_stdout = os.sys.stdout
                os.sys.stdout = open(os.devnull, 'w')
                train_model(model, trainX, trainY, batch_size=batch_size, n_epoch=n_epoch)
                os.sys.stdout = old_stdout
                print("Hidden#1: {0} Hidden#2: {1} Epoch: {2}".format(nunits_hidden1, nunits_hidden2, n_epoch))
                idx = str(nunits_hidden1) + ":" + str(nunits_hidden2)
                results[idx] = test_model(model, test_x, testY)

    print(results)
