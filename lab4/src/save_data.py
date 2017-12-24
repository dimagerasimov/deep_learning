import mxnet as mx
from mxnet import gluon, autograd, ndarray
import numpy as np

import sys
import time

def save_data(file_name, all_data, train_data, encoder, model_ctx):
    start_time = time.time()
    all_data_iterator = gluon.data.DataLoader(all_data, batch_size=1, shuffle=False)
    all_output = mx.nd.zeros((all_data.shape[0], train_data[0][1].size))
    for i, (data) in enumerate(all_data_iterator):
        data = data.as_in_context(model_ctx)
        output = encoder(data)
        all_output[i, :] = output.as_in_context(mx.cpu(0)).reshape(all_output[i, :].shape)
        if i % 50000 == 0:
            print 'Data id = ', i, ' Time: ', time.time() - start_time
            sys.stdout.flush() 
        break


    all_output = all_output.asnumpy()
    print all_output.shape
    np.save(file_name, all_output)

    return

