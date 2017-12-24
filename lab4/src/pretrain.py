import mxnet as mx
from mxnet import gluon, autograd, ndarray
import numpy as np

import sys
import time

def pretrain(train_data, encoder, loss_encoder, decoder, loss_decoder, model_ctx, num_epochs, learning_rate):
    cur_params = gluon.ParameterDict('my_params')
    cur_params.update(encoder.collect_params())
    cur_params.update(decoder.collect_params())

    trainer = gluon.Trainer(cur_params, 'sgd', {'learning_rate': learning_rate})

    epochs = num_epochs
    smoothing_constant = .01

    start_time = time.time()
    for e in range(epochs):
        train_data_shuffle = gluon.data.DataLoader(train_data, batch_size=1, shuffle=True)
        for i, (data, label) in enumerate(train_data_shuffle):
            data = data.as_in_context(model_ctx)
            label = label.as_in_context(model_ctx)        
            with autograd.record():
                hidden = encoder(data)
                dinput = decoder(hidden)
                loss = loss_decoder(dinput, data)
            loss.backward()
            trainer.step(data.shape[0])
            if i % 50000 == 0:
                print 'Data id = ', i, ' Time: ', time.time() - start_time
                sys.stdout.flush() 
            ##########################
            #  Keep a moving average of the losses
            ##########################
            curr_loss = mx.nd.mean(loss).asscalar()
            moving_loss = (curr_loss if ((i == 0) and (e == 0))
                           else (1 - smoothing_constant) * moving_loss + smoothing_constant * curr_loss)

    return encoder, decoder
