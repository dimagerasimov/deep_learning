import numpy as np

def save_data(predictions, id):
	X = np.save('predictions_' + str(id) + '.npy', predictions)

