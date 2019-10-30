'''
Alex Qi Song 09-25-2019
alexsong@vt.edu
usage: python cell_type_classifier.py model_type batch_size pretrain_model_file data_matrix_train_file data_matrix_test_file meta_data_train_file meta_data_test_file output_file
model_type can be "triplet", "constrastive" or "base_nn"
To enable distributed training, make sure you have installed IBM powerAI package, which enable the use of ddl program to launch distributed deep neural network training. See here for detailed examples: https://developer.ibm.com/linuxonpower/2018/09/19/distribute-tensorflow-keras-training-ddl/
'''

from tensorflow.python import keras as keras
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.metrics import binary_accuracy
from tensorflow.python.keras.layers import Input,Dense,Lambda,BatchNormalization,Dropout
from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.losses import categorical_crossentropy

import tensorflow as tf
import numpy as np
import ddl

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA # PCA as accuracy performance baseline model

from pandas import read_csv
from time import time
from subprocess import run,PIPE

from utils import evaluate_embeddings
from utils import mean_average_precision
from utils import get_cell_type_map

import sys
import re

'''
Helper function for Lambda layer
'''
def dist(x):
	a, b = x
	return K.sqrt(K.maximum(K.sum(K.square(a - b), axis=1), K.epsilon()))

'''
Helper function for Lambda layer
'''
def max_indices(x):
	return tf.nn.top_k(x, 1).indices[:,0]

'''
Build a base NN
'''
def build_nn(hidden_layer_sizes, input_dim, l1_reg, l2_reg, activation_func='tanh'):
    
	input_layer = Input(shape=(input_dim,))
	x = input_layer

	# Fully connected hidden layers
	for i, size in enumerate(hidden_layer_sizes):
		x = Dense(size, activation = activation_func, kernel_regularizer=regularizers.l1_l2(l1 = l1_reg, l2 = l2_reg))(x)
		x = Dropout(0.3)(x) if i != len(hidden_layer_sizes) -1 else x 
		x = BatchNormalization()(x)       

	# Wrap base_nn into a model
	base_nn = Model(inputs = input_layer, outputs = x, name = "base_nn")
	return(base_nn, x)

def build_multi_task_nn(base_nn, input_dim, n_cell_types, activation_func = 'softmax'):
    
	X = Input(shape = (input_dim,))
	y = base_nn(X)
    
	# Add a last layer.
	y = Dense(n_cell_types, activation = activation_func)(y)
    
	model = Model(X, y)
    
	return(model)

def get_contrastive_loss(margin):
	def contrastive_loss(y_true, y_pred):

		# Euclidean dist between all pairs
		dist = K.expand_dims(y_pred, axis=1) - K.expand_dims(y_pred, axis=0)
		dist_mat = K.cast(K.sqrt(K.sum(K.square(dist), axis=-1) + K.epsilon()), dtype = 'float32')
		self_mask = K.cast(K.equal(K.expand_dims(y_true, axis=1), K.expand_dims(y_true, axis=0)), dtype = 'float32')

		# Reverse the the positive mask
		neg_mask = K.cast(tf.abs(self_mask - 1), dtype = 'float32')

		# Make the sample do not match with itself
		diag = tf.linalg.diag_part(self_mask) - tf.linalg.diag_part(self_mask)
		pos_mask = K.cast(tf.linalg.set_diag(self_mask,diag), dtype = 'float32')

		pos_loss = 0.5 * K.square(dist_mat * pos_mask)
		neg_loss = 0.5 * K.square(tf.maximum(K.cast(0,dtype='float32'), margin - dist_mat * neg_mask))
		loss = K.mean(pos_loss + neg_loss)
		return loss
    
	return contrastive_loss

def get_triplet_loss(margin, k):
	def triplet_loss(y_true, y_pred):
		# Euclidean dist between all pairs
		dist = K.expand_dims(y_pred, axis=1) - K.expand_dims(y_pred, axis=0)
		dist_mat = K.cast(K.sqrt(K.sum(K.square(dist), axis=-1) + K.epsilon()), dtype = 'float32')
		self_mask = K.cast(K.equal(K.expand_dims(y_true, axis=1), K.expand_dims(y_true, axis=0)), dtype = 'float32')

		# Reverse the the positive mask
		neg_mask = K.cast(tf.abs(self_mask - 1), dtype = 'float32')

		# Make the sample do not match with itself
		diag = tf.linalg.diag_part(self_mask) - tf.linalg.diag_part(self_mask)
		pos_mask = K.cast(tf.linalg.set_diag(self_mask,diag), dtype = 'float32')
		
		# Pick the top K pairs for each positive/negative example(furthest positives and closest negatives)
		top_k_pos = tf.nn.top_k(dist_mat*pos_mask, k).values
		top_k_neg = tf.abs(tf.nn.top_k(-1*(dist_mat*neg_mask + 1e10*self_mask), k).values)

		loss = K.mean(margin + K.expand_dims(top_k_pos,axis = -1) - K.expand_dims(top_k_neg,axis = -2))
		loss = K.maximum(loss,0)
		return loss
	    
	return triplet_loss

# Specify hyperparameters:
learning_rate = 0.01
momentum=0.9
decay=0.01
epochs = 50
k = 10
l1_reg = 0.1
l2_reg = 0

# Get input arguments
model_type,batch_size,pretrain_model,data_train_file,data_test_file,meta_train_file,meta_test_file,out_file = sys.argv[1:]

# Process input arguments and read input files
batch_size = np.int(batch_size)
#pretrain_model = load_model(pretrain_model)
#pretrain_weights = [layer.get_weights() for layer in pretrain_model.layers[1:-1]]
#del pretrain_model
x_train = read_csv(data_train_file, index_col = 0, header = None).values.T
x_test =  read_csv(data_test_file, index_col = 0, header = None).values.T
y_train = read_csv(meta_train_file, header = 0, index_col = 0)
y_test = read_csv(meta_test_file, header = 0, index_col = 0)
label_names = y_train.loc[:,"cell_type"].unique()
label_ids = range(label_names.shape[0])
name_id_map = {key:val for key,val in zip(label_names,label_ids)}
y_train.replace(to_replace = name_id_map,inplace = True)
y_test.replace(to_replace = name_id_map,inplace = True)
y_train = np.array(y_train.loc[:,"cell_type"], dtype = "int64")
y_test = np.array(y_test.loc[:,"cell_type"], dtype = "int64")
n_cell_types = label_names.shape[0]

# DDL: Add the DDL and LMS callback.
callbacks = list()
callbacks.append(ddl.DDLCallback())
callbacks.append(ddl.DDLGlobalVariablesCallback())

# Normalize into 0~1 range
x_train /= x_train.max()
x_test /= x_test.max()

y_train_binary = to_categorical(y_train) # For categorical crossentropy loss, we need to binarize multi-class labels
y_test_binary = to_categorical(y_test)   # For categorical crossentropy loss, we need to binarize multi-class labels

# Split the training data into ddl.size() batches for distributed training.
x_train_dist = np.array_split(x_train, ddl.size())[ddl.rank()]
y_train_dist = np.array_split(y_train, ddl.size())[ddl.rank()]
y_train_dist_binary = np.array_split(y_train_binary, ddl.size())[ddl.rank()]

'''
Training step one. Train for NN
'''
if model_type == 'triplet' or model_type == 'contrastive':
	model = build_nn([568,256,100], x_train_dist.shape[1], l1_reg = l1_reg, l2_reg = l2_reg, activation_func='tanh')[0]
	
	# Set initial weights as DAE trained weights (skip dropout and batchnorm layers)
	# for layer,weight in zip(model.layers[1:8:3],pretrain_weights):
	#	layer.set_weights(weight)
	
	if model_type == 'triplet':
		loss = get_triplet_loss(margin=0.5, k=k)
	else:
		loss = get_contrastive_loss(margin=0.5)


	model.compile(	loss = loss, 
		optimizer = keras.optimizers.SGD(lr=learning_rate * ddl.size(),
		                                    momentum = momentum,
                                            decay = decay
		),
		)
	
	begin = time()
	model.fit(	x_train_dist, 
		y_train_dist,
		batch_size= batch_size,
		epochs = epochs,
		verbose=1 if ddl.rank() == 0 else 0,
		callbacks = callbacks)
	end = time()
	run_time = end-begin

elif model_type == 'base_nn':
	model = build_nn([568,256,100], x_train_dist.shape[1], l1_reg = l1_reg, l2_reg = l2_reg, activation_func='tanh')[0]
	model = build_multi_task_nn(model, input_dim=x_train_dist.shape[1], n_cell_types=n_cell_types)	
	loss = categorical_crossentropy
	# Set initial weights as DAE trained weights (skip dropout and batchnorm layers)
	# for layer,weight in zip(model.layers[1:8:3],pretrain_weights):
	#	layer.set_weights(weight)
	
	begin = time()
	model.compile(	loss = loss, 
		optimizer = keras.optimizers.SGD(lr=learning_rate * ddl.size(),
						momentum = momentum,
						decay = decay		
		),
		)

	model.fit(	x_train_dist, 
			y_train_dist_binary,
			batch_size= batch_size,
			epochs = epochs,
			verbose=1 if ddl.rank() == 0 else 0,
			callbacks = callbacks)
	end = time()
	run_time = end-begin


if ddl.rank() == 0:
	if model_type == "triplet" or model_type == "contrastive":
		train_embedings = model.predict(x_train)
		test_embedings = model.predict(x_test)	
		accuracy, overall_map, cell_type_map= evaluate_embeddings(train_embedings, test_embedings, y_train, y_test, k=50, n_cell_types = n_cell_types)

	elif model_type == "base_nn":
		all_probs = model.predict(x_test)
		order = np.argsort(-all_probs,axis = -1)
		y_pred = order[:,0]
		y_probs = np.array([all_probs[i,order[i,0]] for i in range(all_probs.shape[0])])
		                                                                          
		accuracy = accuracy_score(y_test, y_pred)
		overall_map = mean_average_precision(y_test, y_pred, y_probs)
		cell_type_map = get_cell_type_map(n_cell_types, y_test, y_pred, y_probs)
		np.savetxt("test_label",y_test)
		np.savetxt("pred_label",y_pred)
	with open(out_file,"w") as out_file:
		out_file.writelines('Model type: ' + model_type + "\n")
		out_file.writelines('Number of epochs: ' + str(epochs) + "\n")
		out_file.writelines('Regularization: l1 ' + str(l1_reg) + " l2 " + str(l2_reg) + "\n")
		out_file.writelines('Test accuracy: '+str(accuracy)+"\n")
		out_file.writelines('Overall MAP: '+str(overall_map)+ '\n')
		out_file.writelines('Cell type specific MAP:\n')
		out_file.writelines('\t\t\t\t\t'+'\t'.join(label_names) + '\n')
		out_file.writelines('\t\t\t\t\t'+'\t'.join(map(str,cell_type_map)) + '\n')
		out_file.writelines('Trainning time: '+str(run_time)+" secs\n")
		out_file.writelines('Number of gpus: '+str(ddl.size())+"\n")
		out_file.writelines('Number of nodes: '+str(int(ddl.size()/4))+"\n")
