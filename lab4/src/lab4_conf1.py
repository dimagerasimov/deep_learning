import sys
sys.path.insert(0, "/common/home/deeplearning/studdocs/gerasimov_d/LAB2/mxnet-cuda")

import mxnet as mx
from mxnet import gluon, autograd, ndarray
import numpy as np

import time
start_time = time.time()

from load_data3 import load_data
all_data, train_data, test_data = load_data()

elapsed_time = time.time() - start_time
print 'Time load data: ', elapsed_time

start_time = time.time()

model_ctx = mx.gpu(0)

shape_input = train_data[0][0].shape
size_inputs = train_data[0][0].size
num_outputs = train_data[0][1].size
num_hidden = train_data[0][1].size

from autoencoder_conf1 import prepare_autoencoder1
encoder, loss_encoder, decoder, loss_decoder = prepare_autoencoder1(num_outputs, model_ctx)

num_epochs = 1
learning_rate = .01
from train_autoencoder import train_autoencoder
train_autoencoder(train_data, encoder, loss_encoder, decoder, loss_decoder, model_ctx, num_epochs, learning_rate)

