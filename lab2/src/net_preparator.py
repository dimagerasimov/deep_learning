import mxnet as mx
from mxnet import gluon, autograd, ndarray
import numpy as np

def prepare_net1(num_hidden, num_outputs, model_ctx):
    net = gluon.nn.Sequential()
    with net.name_scope():
        net.add(gluon.nn.Dense(num_hidden, activation="relu", flatten = False))
        print "Activation function in hidden layer 1: relu"
        net.add(gluon.nn.Dense(num_outputs, flatten = False))

    net.collect_params().initialize(mx.init.Normal(sigma=.1), ctx=model_ctx)

    loss_function = gluon.loss.LogisticLoss()
    print "Loss function: LogisticLoss"
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': .07})
    print "Learning rate: 0.07\n"
    return net, loss_function, trainer

def prepare_net2(num_hidden, num_outputs, model_ctx):
    net = gluon.nn.Sequential()
    with net.name_scope():
        net.add(gluon.nn.Dense(num_hidden, activation="relu", flatten = False))
        print "Activation function in hidden layer 1: relu"
        net.add(gluon.nn.Dense(num_hidden, activation="tanh", flatten = False))
        print "Activation function in hidden layer 2: tanh"
        net.add(gluon.nn.Dense(num_outputs, flatten = False))

    net.collect_params().initialize(mx.init.Normal(sigma=.1), ctx=model_ctx)

    loss_function = gluon.loss.HingeLoss()
    print "Loss function: HingeLoss"
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': .05})
    print "Learning rate: 0.05\n"
    return net, loss_function, trainer

def prepare_net3(num_hidden, num_outputs, model_ctx):
    net = gluon.nn.Sequential()
    with net.name_scope():
        net.add(gluon.nn.Dense(num_hidden, activation="tanh", flatten = False))
        print "Activation function in hidden layer 1: tanh"
        net.add(gluon.nn.Dense(num_hidden, activation="sigmoid", flatten = False))
        print "Activation function in hidden layer 2: sigmoid"
        net.add(gluon.nn.Dense(num_outputs, flatten = False))

    net.collect_params().initialize(mx.init.Normal(sigma=.1), ctx=model_ctx)

    loss_function = gluon.loss.SquaredHingeLoss()
    print "Loss function: SquaredHingeLoss"
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': .05})
    print "Learning rate: 0.05\n"
    return net, loss_function, trainer

