import mxnet as mx
from mxnet import gluon, autograd, ndarray
import numpy as np

def prepare_autoencoder1(num_outputs, model_ctx):
    encoder = gluon.nn.Sequential()
    with encoder.name_scope():
        encoder.add(gluon.nn.Conv3D(channels = 1, kernel_size = (7, 3, 3), strides = 1, layout = 'NCDHW'))
        encoder.add(gluon.nn.Flatten())
        encoder.add(gluon.nn.Dense(num_outputs, flatten = False))

    decoder = gluon.nn.Sequential()
    with decoder.name_scope():
        decoder.add(gluon.nn.Dense(294L * 3L * 3L, flatten = False))
        decoder.add(gluon.nn.Lambda(lambda x: mx.nd.reshape(x, shape = (1L, 1L, 294L, 3L, 3L))))
        decoder.add(gluon.nn.Conv3DTranspose(channels = 1, kernel_size = (7, 3, 3), strides = 1, layout = 'NCDHW'))

    encoder.collect_params().initialize(mx.init.Normal(sigma=.1), ctx=model_ctx)
    decoder.collect_params().initialize(mx.init.Normal(sigma=.1), ctx=model_ctx)

    loss_encoder = gluon.loss.LogisticLoss()
    loss_decoder = gluon.loss.L2Loss()

    return encoder, loss_encoder, decoder, loss_decoder
