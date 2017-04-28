"""
    Theano example implementing logistic regression
"""
import os
import theano
import theano.tensor as T
import numpy

#pylint: disable=C0325
#pylint: disable=E1103

def pause():
    """
        pausing until a keystroke
    """
    os.system("pause")

def theano_logistic(samples_size=400, feats=784, verbose_print=False,
                    training_steps=10000, learning_rate=0.1,
                    prediction_threshold=0.6, lambda2_regularization=0.01):
    """
        this function implements logistic regression model
    """
    print("This is example of Theano usage.")
    print("Number of samples: {0}".format(samples_size))
    print("Number of features: {}".format(feats))
    print("Number of training steps: {}".format(training_steps))
    print("Learning rate: {}".format(learning_rate))
    print("Prediction threshold: {}".format(prediction_threshold))
    print("lambda: {}".format(lambda2_regularization))
    pause()

    # generate a dataset: data = (input_values, target_class)
    # TODO - need to modify the dataset
    # TODO - input and target values cannot be 100% random
    # TODO - one need to split the dataset into training/validation/test data
    data = (numpy.random.randn(samples_size, feats),\
        numpy.random.randint(size=samples_size, low=0, high=2))
    # Weight vector initialized randomly
    weights = theano.shared(numpy.random.randn(feats), name="weights")
    # Bias term initialized as 0
    bias = theano.shared(0., name="bias")

    if verbose_print:
        print("Initial model")
        print("Initial weights (random ones)...")
        print(weights.get_value())
        print("Initial bias (zeroed)...")
        print(bias.get_value())
        pause()

    # Construct Theano expression graph
    # Declare Theano symbolic variables
    x = T.dmatrix("x")
    y = T.dvector("y")
    print("... calculation of logit...")
    logit = T.nnet.sigmoid(-T.dot(x, weights) - bias)   # Probability that target = 1
    print("... setting prediction threshold...")
    prediction = logit > prediction_threshold
    print("... calculation of cross-entropy loss function...")
    crossentropyloss = -y * T.log(logit) - (1-y) * T.log(1-logit)
    print("... defining cost function to be minimized...")
    cost = crossentropyloss.mean() + lambda2_regularization*(weights**2).sum()# The cost to minimize
    print("... calculation of gradients for weights and bias:")
    grad_weights, grad_bias = T.grad(cost, [weights, bias])

    print("... definition of model...")
    train = theano.function(
        inputs=[x, y], outputs=[prediction, crossentropyloss],
        updates=((weights, weights - learning_rate * grad_weights),
                 (bias, bias - learning_rate * grad_bias)))
    print("... running predictions based on the trained model...")
    predict = theano.function(inputs=[x], outputs=prediction)

    print("... training the model...")
    for _ in range(training_steps):
        # TODO - show the error
        #pred, err = train(data[0], data[1])
        train(data[0], data[1])

    if verbose_print:
        print("Final model:")
        print(weights.get_value())
        print(bias.get_value())

    print("target values for data:")
    print(data[1])
    print("prediction on data:")
    print(predict(data[0]))

theano_logistic(training_steps=5000)
