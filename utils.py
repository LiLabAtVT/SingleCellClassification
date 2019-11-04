import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score

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

	# Use the similarity as weights. Count occurances of each cell type among the kompute overall mean average precisions (MAP)
	knn_sim_cell_type = np.sum(knn_sim_3d * knn_labels_one_hot, axis = 1)

	# Compute accuracy
	pred_labels = np.argmax(knn_sim_cell_type, axis = 1)
	accuracy = accuracy_score(test_labels, pred_labels)

	# Compute overall mean average precisions (MAP)
	pred_decision_val = np.max(knn_sim_cell_type, axis = 1)
	overall_map = mean_average_precision(test_labels, pred_labels, pred_decision_val)

	# Compute cell-type-specific MAP
	cell_type_map = get_cell_type_map(n_cell_types, test_labels, pred_labels, pred_decision_val)
	
	return(accuracy, overall_map, cell_type_map)

def mean_average_precision(test_labels, pred_labels, pred_decision_val):
    
	order = np.argsort(-pred_decision_val)
	test_labels_reord = test_labels[order]
	pred_labels_reord = pred_labels[order]
	precision_list = np.zeros((test_labels.shape[0]))
	for i in range(test_labels.shape[0]):
		precision_list[i] = np.sum(test_labels_reord[:i+1] == pred_labels_reord[:i+1])/np.float32(i+1)
    
	return np.mean(precision_list)

# Compute cell-type-specific MAP
def get_cell_type_map(n_cell_types, test_labels, pred_labels,pred_decision_val):
	cell_type_map = np.zeros((n_cell_types))
	for i,label in enumerate(np.unique(test_labels)):
		mask = (test_labels == label)
		cell_type_map[i] = mean_average_precision(test_labels[mask], pred_labels[mask], pred_decision_val[mask])
	return(cell_type_map)