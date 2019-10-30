# Single Cell Classification
## 1. This Repository
This repository houses several scripts for performing single cell classification using machine learning methods. Neural network based classification was performed with Keras and all other types of classification were performed with scikit-learn.
## 2. Files
### 2.1 Script Files
- cell_type_classifier_NN_ddl.py: Neural network based classification using triplet, contrastive, and cross entropy as - objective functions. `_ddl` indicates this script should be run with ddlrun on a IBM PowerAI machine, which enables distributed deep learning across multiple GPUs and mutiple computing nodes.

- cell_type_classifier_NN_single.py: Neural network based classification using triplet, contrastive, and cross entropy as objective functions. This script can be run without ddlrun but only trains neural network on a single GPU.

- DAE_pretrain.py: Pretrain a neural network with Denoising Auto Encoder. See [Amir et al., 2018 (https://www.nature.com/articles/s41467-018-07165-2).

- cell_type_classifier_PCA_SVM_RF.ipynb: A python jupyter notebook to perform classification using SVM, RF, KNN, and PCA.
### 2.2 Example Data Files
Example data files are located in VT cluster under /groups/songli_lab/single_cell_analysis/data/ath/for_machine_learning
See README file in the folder for details about each data file
## 3. Usage
Usage info can be found in the comment lines of each script file
## 4. References
https://www.nature.com/articles/s41467-018-07165-2
