import mxnet as mx
from mxnet import gluon, autograd, ndarray
import numpy as np

def load_data():
	X = np.load('data/X.npy')
	Y = np.load('data/Y.npy')
	print type(X[0,0])
	Y = (Y / 189.0) * 2.0 - 1.0
	n = 512
	m = 512
	WINDOW_HALFSIZE = 2
	WINDOW_SIZE = WINDOW_HALFSIZE * 2 + 1
	Xn = np.zeros((X.shape[0], 1, X.shape[1], WINDOW_SIZE, WINDOW_SIZE), dtype = 'float32')
	
	for i in range(0, n):
		for j in range(0, m):
			for dx in range(-WINDOW_HALFSIZE, WINDOW_HALFSIZE + 1):
				for dy in range(-WINDOW_HALFSIZE, WINDOW_HALFSIZE + 1):
					ii = i + dx
					jj = j + dy
					if ii >= 0 and jj >= 0 and ii < n and jj < m:
						Xn[i * m + j, 0, :, dx + WINDOW_HALFSIZE, dy + WINDOW_HALFSIZE] = X[ii * m + jj, :]
	
	from sklearn.model_selection import train_test_split
	X_train, X_test, Y_train, Y_test = train_test_split(Xn, Y, train_size = 0.66 / 2.0, test_size = 0.33 / 2.0, random_state=42)
	print X_train.nbytes, ' ', X_test.nbytes
	
	train_data = mx.gluon.data.ArrayDataset(mx.nd.array(X_train), mx.nd.array(Y_train))
	test_data = mx.gluon.data.ArrayDataset(mx.nd.array(X_test), mx.nd.array(Y_test))

	return train_data, test_data

