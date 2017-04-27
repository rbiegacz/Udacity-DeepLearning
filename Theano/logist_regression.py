"""
    Theano example implementing logistic regression
"""
import os
import theano
import theano.tensor as T
import numpy

#pylint: disable = C0325
#pylint: disable=E1103

def pause():
    """
        pausing until a keystroke
    """
    os.system("pause")

def theano_logistic(samples_size=400, feats=784, verbose_print=False,
                    training_steps=10000, learning_rate=0.1,
                    prediction_threshold=0.6, Lambda=0.01):
    """
        theano_logistic implements a Theano model
    """
    print("This is example of Theano usage.")
    print"Number of samples: {}".format(samples_size)
    print "Number of features: {}".format(feats)
    print "Number of training steps: {}".format(training_steps)
    print "Learning rate: {}".format(learning_rate)
    print "Prediction threshold: {}".format(prediction_threshold)
    print "lambda: {}".format(Lambda)
    pause()

    # generate a dataset: D = (input_values, target_class)
    rng = numpy.random
    D = (numpy.random.randn(samples_size, feats),\
        numpy.random.randint(size=samples_size, low=0, high=2))
    # Weight vector initialized randomly
    w = theano.shared(rng.randn(feats), name="w")
    # Bias term initialized as 0
    b = theano.shared(0., name="b")

    if verbose_print:
        print "Initial model"
        print "Initial weights (random ones)..."
        print w.get_value()
        print "Initial bias (zeroed)..."
        print b.get_value()
        pause()

    # Construct Theano expression graph
    # Declare Theano symbolic variables
    x = T.dmatrix("x")
    y = T.dvector("y")
    print("... calculation of logit...")
    logit = T.nnet.sigmoid(-T.dot(x, w) - b)   # Probability that target = 1
    print("... setting prediction threshold...")
    prediction = logit > prediction_threshold
    print("... calculation of cross-entropy loss function...")
    xent = -y * T.log(logit) - (1-y) * T.log(1-logit)
    print("... defining cost function to be minimized...")
    cost = xent.mean() + Lambda * (w ** 2).sum()# The cost to minimize
    print("... calculation of gradients for weights and bias:")
    gw, gb = T.grad(cost, [w, b])

    print("... definition of model...")
    train = theano.function(
        inputs=[x, y], outputs=[prediction, xent],
        updates=((w, w - learning_rate * gw), (b, b - learning_rate * gb)))
    print("... running predictions based on the trained model...")
    predict = theano.function(inputs=[x], outputs=prediction)

    print("... training the model...")
    for _ in range(training_steps):
        #pred, err = train(D[0], D[1])
        train(D[0], D[1])

    if verbose_print:
        print("Final model:")
        print(w.get_value())
        print(b.get_value())

    print("target values for D:")
    print(D[1])
    print("prediction on D:")
    print(predict(D[0]))

theano_logistic()
