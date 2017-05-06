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
    train_x_data, train_y_data, test_x_data, test_y_data = mnist.load_data(one_hot=True)
    return train_x_data, train_y_data, test_x_data, test_y_data

def build_model(nunits_hidden1=128, nunits_hidden2=28, learning_rate_value=0.1):
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
                            learning_rate=learning_rate_value, loss='categorical_crossentropy')
    return tflearn.DNN(net)

def train_model(modl, train_x_data, train_y_data, batch_size_number=50, number_of_epochs=20):
    """
        this function trains the model
    """
    return modl.fit(train_x_data, train_y_data, validation_set=0.1,\
                    show_metric=True, batch_size=batch_size_number, n_epoch=number_of_epochs)

def test_model(mod, text_x, text_y):
    """
        this function tests the model
    """
    predictions = np.array(mod.predict(text_x)).argmax(axis=1)
    actual = text_y.argmax(axis=1)
    test_accuracy = np.mean(predictions == actual, axis=0)
    print("Test accuracy: ", test_accuracy)
    return test_accuracy

if __name__ == "__main__":
    batch_size = 50
    n_epochs = [50, 100]
    l_rates = [1, 0.1, 0.01]
    nunits_hidden1s = [28, 56, 128, 256, 512]
    nunits_hidden2s = [28, 56, 128, 256, 512]
    results = {}

    train_x, train_y, test_x, testY = load_data()
    for l_rate in l_rates:
        for n_epoch in n_epochs:
            for nhidden1 in nunits_hidden1s:
                for nhidden2 in nunits_hidden2s:
                    if nhidden2 > nhidden1:
                        continue
                    print("Hidden#1:{0} Hidden#2:{1} Epoch:{2} L.Rate:{3}"\
                          .format(nhidden1, nhidden2, n_epoch, l_rate))
                    tf.reset_default_graph()
                    model = build_model(nunits_hidden1=nhidden1, nunits_hidden2=nhidden2,\
                                        learning_rate_value=l_rate)
                    old_stdout = os.sys.stdout
                    os.sys.stdout = open(os.devnull, 'w')
                    train_model(model, train_x, train_y,\
                                batch_size_number=batch_size, number_of_epochs=n_epoch)
                    os.sys.stdout = old_stdout
                    idx = str(nhidden1) + ":" + str(nhidden2)
                    results[idx] = test_model(model, test_x, testY)
    print(results)
    print("The best accuracy: {}".format(max(results, key=results.get)))
