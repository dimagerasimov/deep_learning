import mxnet as mx
from mxnet import gluon, autograd, ndarray
import numpy as np

def load_data():
	X = np.load('data/X.npy')
	Y = np.load('data/Y.npy')
	Y = (Y / 189.0) * 2.0 - 1.0
	#X = X[1:5000, :]
	#Y = Y[1:5000, :]
	from sklearn.model_selection import train_test_split
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
	
	train_data = mx.gluon.data.ArrayDataset(mx.nd.array(X_train), mx.nd.array(Y_train))
	test_data = mx.gluon.data.ArrayDataset(mx.nd.array(X_test), mx.nd.array(Y_test))

	return train_data, test_data

