# usage: python DAE_pretrain.py batch_size data_matrix_train_file out_file
from tensorflow.python import keras as keras
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.metrics import binary_accuracy
from tensorflow.python.keras.layers import Input,Dense
from tensorflow.python.keras.models import Model
from pandas import read_csv

import numpy as np
import sys

# Build an autoencoder. "weights" should be a list of arrays, in which each element is a two-element
# list with first element being an array for weights and second element being an array for
# biases.
def build_DAE(hidden_layer_sizes, input_dim, weights = [], activation_func = 'tanh'):
	
	input_layer = Input(shape=(input_dim,))
	x = input_layer 
	
	# Fully connected hidden layers
	for i, size in enumerate(hidden_layer_sizes):
		x = Dense(size, activation = activation_func)(x)
	
	# Output layer for DAE, which has same number of neurons with the input layer
	y = Dense(input_dim, activation = activation_func)(x)
	model = Model(inputs = input_layer, outputs = y, name = "DAE")

	# Set model weights, if weights are provided
	if len(weights) > 0:
		for layer,weight in zip(model.layers[1:], weights):
			layer.set_weights(weight)	 
	return(model)	

# Specify hyperparameters:
learning_rate = 0.01
momentum=0.9
decay=0.01
epochs = 300

batch_size,data_train_file,out_file = sys.argv[1:]

# Process input arguments and read input files
batch_size = np.int(batch_size)
x_train = read_csv(data_train_file, index_col = 0, header = None).values.T

# Add Guassian noise to data.
x_train_noisy = x_train + np.random.normal(size=x_train.shape)

# Normalize into 0~1 range
x_train_noisy /= x_train_noisy.max()
x_train /= x_train.max()

pretrain_layer_sizes = [568, 256, 100]
weights = []

#  Pretrain weights layer by layer using DAE
for k in range(len(pretrain_layer_sizes)):
	model = build_DAE(pretrain_layer_sizes[:(k+1)], x_train.shape[1], weights = weights, activation_func = 'tanh')
	
	# Freeze the previous layers so that their weights will not be updated
	for j in range(k):
		model.layers[j+1].trainable = False
	
	# Compile model
	model.compile(loss = 'binary_crossentropy',
		optimizer = keras.optimizers.SGD(lr=learning_rate,
						momentum = momentum,
						decay = decay
						)
			)

	# Train DAE
	model.fit(x_train,
		x_train_noisy,
		batch_size = batch_size,
		epochs = epochs,
		verbose=1
		)
	# Get trained weights from the layer before the last layer
	weights.append(model.layers[-2].get_weights())

# Save pretrained model
model.save(out_file)