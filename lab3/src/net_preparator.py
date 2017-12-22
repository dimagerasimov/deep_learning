import mxnet as mx
from mxnet import gluon, autograd, ndarray
import numpy as np

def prepare_net1(num_hidden, num_outputs, model_ctx):
    net = gluon.nn.Sequential()
    with net.name_scope():
        net.add(gluon.nn.Conv3D(channels = 1, kernel_size = (5, 2, 2), strides = 1, layout = 'NCDHW'))
        print "Hidden layer 1: Conv3D WHERE channels = 1, kernel_size = (5, 2, 2), strides = 1, layout = 'NCDHW'"
        net.add(gluon.nn.MaxPool3D(pool_size = 2, strides = 1, layout = 'NCDHW'))
        print "Hidden layer 2: MaxPool3D WHERE pool_size = 2, strides = 1"
        net.add(gluon.nn.Flatten())
        print "Hidden layer 3: Flatten"
        net.add(gluon.nn.Dense(num_hidden, activation="relu", flatten = False))
        print "Activation function in hidden layer 4: relu"
        net.add(gluon.nn.Dense(num_outputs, flatten = False))

    net.collect_params().initialize(mx.init.Normal(sigma=.1), ctx=model_ctx)

    loss_function = gluon.loss.LogisticLoss()
    print "Loss function: LogisticLoss"
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': .007})
    print "Learning rate: 0.007\n"
    return net, loss_function, trainer

def prepare_net2(num_hidden, num_outputs, model_ctx):
    net = gluon.nn.Sequential()
    with net.name_scope():
        net.add(gluon.nn.Conv3D(channels = 2, kernel_size = (10, 3, 3), strides = 1, layout = 'NCDHW'))
        print "Hidden layer 1: Conv3D WHERE channels = 2, kernel_size = (10, 3, 3), strides = 1, layout = 'NCDHW'"
        net.add(gluon.nn.AvgPool3D(pool_size = 2, strides = 1, layout = 'NCDHW'))
        print "Hidden layer 2: AvgPool3D WHERE pool_size = 2, strides = 1"
        net.add(gluon.nn.Flatten())
        print "Hidden layer 3: Flatten"
        net.add(gluon.nn.Dense(num_hidden, activation="tanh", flatten = False))
        print "Activation function in hidden layer 4: tanh"
        net.add(gluon.nn.Dense(num_outputs, flatten = False))

    net.collect_params().initialize(mx.init.Normal(sigma=.1), ctx=model_ctx)

    loss_function = gluon.loss.LogisticLoss()
    print "Loss function: LogisticLoss"
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': .007})
    print "Learning rate: 0.007\n"
    return net, loss_function, trainer

def prepare_net3(num_hidden, num_outputs, model_ctx):
    net = gluon.nn.Sequential()
    with net.name_scope():
        net.add(gluon.nn.Conv3D(channels = 1, kernel_size = (7, 3, 3), strides = 1, layout = 'NCDHW'))
        print "Hidden layer 1: Conv3D WHERE channels = 1, kernel_size = (7, 3, 3), strides = 1, layout = 'NCDHW'"
        net.add(gluon.nn.MaxPool3D(pool_size = 2, strides = 1, layout = 'NCDHW'))
        print "Hidden layer 2: MaxPool3D WHERE pool_size = 2, strides = 1"
        net.add(gluon.nn.Flatten())
        print "Hidden layer 3: Flatten"
        net.add(gluon.nn.Dense(num_hidden, activation="relu", flatten = False))
        print "Activation function in hidden layer 4: relu"
        net.add(gluon.nn.Dense(num_outputs, flatten = False))

    net.collect_params().initialize(mx.init.Normal(sigma=.1), ctx=model_ctx)

    loss_function = gluon.loss.HingeLoss()
    print "Loss function: HingeLoss()"
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': .007})
    print "Learning rate: 0.007\n"
    return net, loss_function, trainer

