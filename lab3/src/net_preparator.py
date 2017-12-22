import mxnet as mx
from mxnet import gluon, autograd, ndarray
import numpy as np

def prepare_net1(num_hidden, num_outputs, model_ctx):
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

    loss_function = gluon.loss.LogisticLoss()
    print "Loss function: LogisticLoss"
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': .01})
    print "Learning rate: 0.01\n"
    return net, loss_function, trainer
