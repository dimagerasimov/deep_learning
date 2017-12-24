import mxnet as mx
from mxnet import gluon, autograd, ndarray
import numpy as np

import sys
import time

def evaluate_accuracy(data_iterator, net, start_time, model_ctx):
    acc = 0
    cnt = 0.0
    data_iterator = gluon.data.DataLoader(data_iterator, batch_size=1, shuffle=False)
    for i, (data, label) in enumerate(data_iterator):
        data_ctx = data.as_in_context(model_ctx)
        label_ctx = label.as_in_context(model_ctx)
        output = net(data_ctx)
        output = output.as_in_context(mx.cpu(0))
        if i == 0:
            print (output)
            print (label)
        sq = (output * label) > 0
        acc += sq.mean()
        cnt += 1
        if i % 1000 == 0:
            print 'Metric. Data id = ', i, ' Time: ', time.time() - start_time
            sys.stdout.flush()
        break
    return (acc.asscalar() / cnt)

def train(train_data, test_data, encoder, loss_encoder, decoder, loss_decoder, model_ctx, num_epochs, learning_rate):
    cur_params = gluon.ParameterDict('my_params')
    cur_params.update(encoder.collect_params())

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
                print data.shape
                output = encoder(data)
                print output.shape
                loss = loss_encoder(output, label)
            loss.backward()
            trainer.step(data.shape[0])
            if i % 1000 == 0:
                print 'Data id = ', i, ' Time: ', time.time() - start_time
                sys.stdout.flush() 
            ##########################
            #  Keep a moving average of the losses
            ##########################
            curr_loss = mx.nd.mean(loss).asscalar()
            moving_loss = (curr_loss if ((i == 0) and (e == 0))
                           else (1 - smoothing_constant) * moving_loss + smoothing_constant * curr_loss)
            break

        test_accuracy = evaluate_accuracy(test_data, encoder, start_time, model_ctx)
        train_accuracy = evaluate_accuracy(train_data, encoder, start_time, model_ctx)
        print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" % (e, moving_loss, train_accuracy, test_accuracy))

    return

