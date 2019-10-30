'''
Alex Qi Song 09-25-2019
alexsong@vt.edu
python cell_type_classifier.py model_type batch_size data_matrix_train_file data_matrix_test_file meta_data_train_file meta_data_test_file output_file
model_type can be "triplet", "contrastive" or "base_nn"
'''

from tensorflow.python import keras as keras
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.metrics import binary_accuracy
from tensorflow.python.keras.layers import Input,Dense,Lambda,BatchNormalization,Dropout
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.losses import categorical_crossentropy

import tensorflow as tf
import numpy as np

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA # PCA as accuracy performance baseline model

from pandas import read_csv
from time import time

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
def build_nn(hidden_layer_sizes, input_dim, l1_reg, activation_func='relu'):
	input_layer = Input(shape=(input_dim,))
	x = input_layer
	x = Dropout(0.3, input_shape=(input_dim,))(x)

	# Fully connected hidden layers
	for i, size in enumerate(hidden_layer_sizes):
		x = Dense(size, activation = activation_func, kernel_regularizer=regularizers.l1(l1_reg))(x)
		x = Dropout(0.1)(x)        
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
	
def evaluate_embeddings(train_embedings, test_embedings, train_labels, test_labels, k, n_cell_types):
    
	# Pairwise euclidean distance between test_embedings and train_embedings
	dist = np.expand_dims(test_embedings, axis = 1) - np.expand_dims(train_embedings, axis = 0)
	dist_mat = np.sqrt(np.sum(np.square(dist), axis=-1))

	# Pick k nearest neighbors for each test example
	knn_indices = np.argsort(dist_mat)[:,:k]
	knn_dist = np.vstack([dist_mat[i, knn_indices[i,:]] for i in range(knn_indices.shape[0])])
	knn_labels = np.vstack([train_labels[knn_indices[i, :]] for i in range(knn_indices.shape[0])])

	# Convert distance to similarity
	knn_sim_2d = 1/(1 + knn_dist)
	knn_sim_3d = np.expand_dims(knn_sim_2d,axis = -1)

	# One-hot array to count occurances of each cell type among the k nearst neighbors
	knn_labels_one_hot = LabelBinarizer().fit_transform(knn_labels.flatten())
	knn_labels_one_hot = knn_labels_one_hot.reshape((knn_labels.shape[0], knn_labels.shape[1], -1))

	# Use the similarity as weights. Count occurances of each cell type among the k nearst neighbors
	knn_sim_cell_type = np.sum(knn_sim_3d * knn_labels_one_hot, axis = 1)

	# Compute accuracy
	pred_labels = np.argmax(knn_sim_cell_type, axis = 1)
	accuracy = accuracy_score(test_labels, pred_labels)

	# Compute overall mean average precisions (MAP)
	pred_decision_val = np.max(knn_sim_cell_type, axis = 1)
	overall_map = mean_average_precision(test_labels, pred_labels, pred_decision_val)

    	# Compute cell-type-specific MAP
	cell_type_map = np.zeros((n_cell_types))
	for i,label in enumerate(np.unique(test_labels)):
		mask = (test_labels == label)
		cell_type_map[i] = mean_average_precision(test_labels[mask], pred_labels[mask], pred_decision_val[mask])

	return(accuracy, overall_map, cell_type_map)

def mean_average_precision(test_labels, pred_labels, pred_decision_val):
    
	order = np.argsort(-pred_decision_val)
	test_labels_reord = test_labels[order]
	pred_labels_reord = pred_labels[order]
	precision_list = np.zeros((test_labels.shape[0]))
	for i in range(test_labels.shape[0]):
		precision_list[i] = np.sum(test_labels_reord[:i+1] == pred_labels_reord[:i+1])/np.float32(i+1)

	return np.mean(precision_list)

# Specify hyperparameters:
test_size = 0.2
learning_rate = 0.01
epochs = 50
k = 20
l1_reg = 100

# Get input arguments
model_type,batch_size,data_train_file,data_test_file,meta_train_file,meta_test_file,out_file = sys.argv[1:]

batch_size = np.int(batch_size)
x_train = read_csv(data_train_file,sep = "\t", index_col = 0, header = None).values.T
x_test =  read_csv(data_test_file,sep = "\t", index_col = 0, header = None).values.T
y_train = read_csv(meta_train_file,sep = "\t", header = 0, index_col = 0)
y_test = read_csv(meta_test_file,sep = "\t", header = 0, index_col = 0)
label_names = y_train.loc[:,"cell_type"].unique()
label_ids = range(label_names.shape[0])
name_id_map = {key:val for key,val in zip(label_names,label_ids)}
y_train.replace(to_replace = name_id_map,inplace = True)
y_test.replace(to_replace = name_id_map,inplace = True)
y_train = np.array(y_train.loc[:,"cell_type"], dtype = "int64")
y_test = np.array(y_test.loc[:,"cell_type"], dtype = "int64")
n_cell_types = label_names.shape[0]

# Normalize into 0~1 range
x_train /= x_train.max()
x_test /= x_test.max()

y_train_binary = to_categorical(y_train) # For categorical crossentropy loss, we need to binarize multi-class labels
y_test_binary = to_categorical(y_test)   # For categorical crossentropy loss, we need to binarize multi-class labels

'''
Training step one. Train for NN
'''
if model_type == 'triplet' or model_type == 'contrastive':
	model = build_nn([1136,500,100], x_train.shape[1], l1_reg = l1_reg, activation_func='tanh')[0]
	
	if model_type == 'triplet':
		loss = get_triplet_loss(margin=0.5, k=k)
	else:
		loss = get_contrastive_loss(margin=0.5)
	
	model.compile(	loss = loss, 
		optimizer = keras.optimizers.Adam(lr=learning_rate),
		)

	# Training	
	begin = time()
	model.fit(	x_train, 
			y_train,
			batch_size=batch_size,
			epochs = epochs,
			verbose=1
		)
	end = time()
	run_time = end-begin

elif model_type == 'base_nn':
	model = build_nn([1136,500,100], x_train.shape[1], l1_reg = l1_reg, activation_func='tanh')[0]
	model = build_multi_task_nn(model, input_dim=x_train.shape[1], n_cell_types=n_cell_types)	
	loss = categorical_crossentropy
	
	model.compile(	loss = loss, 
		optimizer = keras.optimizers.Adam(lr=learning_rate),
		)

	# Training	
	begin = time()
	model.fit(	x_train, 
			y_train_binary,
			batch_size=batch_size,
			epochs = epochs,
			verbose=1
		)
	end = time()
	run_time = end-begin 


if model_type == "triplet" or model_type == "contrastive":
	train_embedings = model.predict(x_train)
	test_embedings = model.predict(x_test)	
	accuracy, overall_map, cell_type_map= evaluate_embeddings(train_embedings, test_embedings, y_train, y_test, k=20, n_cell_types = n_cell_types)

elif model_type == "base_nn":
	all_probs = model.predict(x_test)
	order = np.argsort(-all_probs,axis = -1)
	y_pred = order[:,0]
	y_probs = np.array([all_probs[i,order[i,0]] for i in range(all_probs.shape[0])])	                                                                  
	accuracy = accuracy_score(y_test, y_pred)
	overall_map = mean_average_precision(y_test, y_pred, y_probs)

with open(out_file,"w") as out_file:
	out_file.writelines('Model type: ' + model_type + "\n")
	out_file.writelines('Test accuracy: '+str(accuracy)+"\n")
	out_file.writelines('Trainning time: '+str(run_time)+" secs\n")
	out_file.writelines('Number of gpus: 1\n')
	out_file.writelines('Number of nodes:1\n')
