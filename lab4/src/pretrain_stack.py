import mxnet as mx
from mxnet import gluon, autograd, ndarray
import numpy as np

import sys
import time

def pretrain_stack(train_data, encoder, loss_encoder, decoder, loss_decoder, model_ctx, num_epochs, learning_rate):
    epochs = num_epochs
    smoothing_constant = .01

    start_time = time.time()
    for layer_id, layer_encoder in enumerate(encoder):
        print layer_id
        print len(decoder)
        layer_decoder = decoder[len(decoder) - layer_id - 1]
        print('layer_encoder', layer_encoder.__dict__['_name'])
        print('layer_decoder', layer_decoder.__dict__['_name'])
        if layer_decoder.__dict__['_name'].find('lambda') != -1:
        	continue
        cur_params = gluon.ParameterDict('my_params')
        cur_params.update(layer_encoder.collect_params())
        cur_params.update(layer_decoder.collect_params())

        trainer = gluon.Trainer(cur_params, 'sgd', {'learning_rate': .01})
        for e in range(epochs):
            train_data_shuffle = gluon.data.DataLoader(train_data, batch_size=1, shuffle=True)
            for i, (data, label) in enumerate(train_data_shuffle):
                data = data.as_in_context(model_ctx)
                label = label.as_in_context(model_ctx)        
                encoded_input = data
                for j in range(0, layer_id):
                    encoded_input = encoder[j](encoded_input)
           
                with autograd.record():
                    encoded_layer = layer_encoder(encoded_input)
                    decoded_input = layer_decoder(encoded_layer)
                    loss = loss_decoder(decoded_input, encoded_input)

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
