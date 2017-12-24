import sys

import mxnet as mx
from mxnet import gluon, autograd, ndarray
import numpy as np
import time

def compute_and_save_output(net, model_ctx, all_data, id):
	start_time = time.time()
	all_data_iterator = gluon.data.DataLoader(all_data, batch_size=1, shuffle=False)
	all_output = mx.nd.zeros(all_data.shape)
	for i, (data) in enumerate(all_data_iterator):
		data = data.as_in_context(model_ctx)
		output = net(data)
		all_output[i, :] = output.reshape(all_output[i, :].shape)
		if i % 10000 == 0:
			print 'Create output id = ', i, ' Time: ', time.time() - start_time
			sys.stdout.flush() 

	all_output = all_output.as_in_context(mx.cpu(0)).asnumpy()
	print all_output.shape
	np.save('./results/predictions/predictions_' + str(id), all_output)
