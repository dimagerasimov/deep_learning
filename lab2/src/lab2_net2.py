import sys
sys.path.insert(0, "/common/home/deeplearning/studdocs/gerasimov_d/LAB2/mxnet-cuda")

import mxnet as mx
from mxnet import gluon, autograd, ndarray
import numpy as np

import time
start_time = time.time()

from load_data import load_data
train_data, test_data = load_data()

elapsed_time = time.time() - start_time
print 'Time load data: ', elapsed_time

start_time = time.time()

model_ctx = mx.gpu(0)

num_outputs = train_data[0][0].size
num_hidden = train_data[0][1].size

from net_preparator import prepare_net2
print "\nPreparing the network"
net, loss_function, trainer = prepare_net2(num_hidden, num_outputs, model_ctx)

def evaluate_accuracy(data_iterator, net):
    acc = 0 #mx.metric.RMSE()
    cnt = 0.0
    for i, (data, label) in enumerate(data_iterator):
        data_ctx = data.as_in_context(model_ctx)
        label_ctx = label.as_in_context(model_ctx)
        output = net(data_ctx)
        output = output.as_in_context(mx.cpu(0))
        #rmse.update(preds=output, labels=label)
        if i == 0:
            print (output)
            print (label)
        sq = (output * label) > 0
        acc += sq.mean()
        cnt += 1
        if i % 50000 == 0:
            print 'Metric. Data id = ', i, ' Time: ', time.time() - start_time
            sys.stdout.flush() 
    return (acc.asscalar() / cnt) #rmse.get()[1]

elapsed_time = time.time() - start_time
print 'Time preparing net: ', elapsed_time
import sys
sys.stdout.flush()

start_time = time.time()

epochs = 3
smoothing_constant = .01

for e in range(epochs):
    train_data_shuffle = gluon.data.DataLoader(train_data, batch_size = 1, shuffle=True)
    for i, (data, label) in enumerate(train_data_shuffle):
        data = data.as_in_context(model_ctx)
        label = label.as_in_context(model_ctx)
        with autograd.record():
            output = net(data)
            loss = loss_function(output, label)
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
		

    test_accuracy = evaluate_accuracy(test_data, net)
    train_accuracy = evaluate_accuracy(train_data, net)
    print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" % (e, moving_loss, train_accuracy, test_accuracy))

elapsed_time = time.time() - start_time
print 'Time preparing net: ', elapsed_time
