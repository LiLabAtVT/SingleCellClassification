#!/usr/bin/env python

##updation 041720 for 02var so we need to add the RF again
##updation 041720 ##add all the evaluation
##updation 041720 ##rm the RF
##updation 041420 ##rm the PCA and KNN
##updation 040720 ##add save to pca and knn models
##updation 040320 ##remove the permuation importance
##updation 022720 ##utilize permutation importance
##updation 022420 ##save the model to the output dir
##updation 022220 ##add save model option and use linear for the SVM
##updation 012820

##this script is to train the PCA SVM RF model using python script

##import modules
##utils is another script used for evaluation

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from pandas import read_csv
from sklearn import svm
import pickle
from sklearn.metrics import classification_report
from sklearn.metrics import multilabel_confusion_matrix
import math


def mean_average_precision(test_labels, pred_labels, pred_decision_val):
    order = np.argsort(-pred_decision_val)
    test_labels_reord = test_labels[order]
    pred_labels_reord = pred_labels[order]
    precision_list = np.zeros((test_labels.shape[0]))
    for i in range(test_labels.shape[0]):
        precision_list[i] = np.sum(test_labels_reord[:i + 1] == pred_labels_reord[:i + 1]) / np.float32(i + 1)

    return np.mean(precision_list)


# Compute cell-type-specific MAP
def get_cell_type_map(n_cell_types, test_labels, pred_labels, pred_decision_val):
    cell_type_map = np.zeros((n_cell_types))
    for i, label in enumerate(np.unique(test_labels)):
        mask = (test_labels == label)
        cell_type_map[i] = mean_average_precision(test_labels[mask], pred_labels[mask], pred_decision_val[mask])
    return (cell_type_map)

##define a function to evaluate
def evaluation (labels_pred,prob_pred,labels_test,label_names,test_type,id_nm_dic,model_name,input_opt_dir):

    ##model_name is svm or rf

    ##test_type is independent or validate

    ##evaluate predicted cell types for svm, rf and knn
    acc = []
    overall_map = []
    cell_type_map = []
    for y_pred, y_prob_pred, y_test in zip(labels_pred, prob_pred, np.tile(labels_test,(3,1))):
        acc.append(accuracy_score(y_test, y_pred))
        overall_map.append(mean_average_precision(y_test, y_pred, y_prob_pred))
        cell_type_map.append(get_cell_type_map(label_names.shape[0], y_test, y_pred, y_prob_pred))

    ###########################
    ##write map results for knn
    ##initiate a dic to store method id and its name
    store_method_id_nm = {'1':model_name}
    meth_id = 0

    for eachpred_method in cell_type_map:
        meth_id += 1
        ##write name of method
        meth_nm = store_method_id_nm[str(meth_id)]
        single_cell_id = -1
        with open(input_opt_dir + '/opt_' + meth_nm + '_' + test_type + '_results_multiple_class.txt', 'w+') as opt:
            for eachvalue in eachpred_method:
                single_cell_id += 1
                single_cell_name = id_nm_dic[str(single_cell_id)]
                final_line = str(single_cell_id) + '\t' + single_cell_name + '\t' + str(eachvalue)
                opt.write(final_line + '\n')

    ################################
    ##write other evaluation reports
    target_names = []
    for i in range(len(list(id_nm_dic.keys()))):
        target_names.append(id_nm_dic[str(i)])
    with open(input_opt_dir + '/opt_' + model_name + '_' + test_type + '_class_report.txt', 'w+') as opt:
        opt.write(classification_report(labels_test, labels_pred[0],
                                        target_names=target_names) + '\n')  ##labels_pred is a list

    ##calculate TP TN FP FN
    mcm = multilabel_confusion_matrix(labels_test, labels_pred[0])
    tn = mcm[:, 0, 0]
    tp = mcm[:, 1, 1]
    fn = mcm[:, 1, 0]
    fp = mcm[:, 0, 1]

    SE = tp / (tp + fn)
    AC = (tp + tn) / (tp + tn + fp + fn)
    PR = tp / (tp + fp)
    SP = tn / (fp + tn)
    ##MCC and GM
    MCC_list = []
    GM_list = []
    for i in range(len(AC)):
        s_tp = tp[i]  ##single tp
        s_tn = tn[i]
        s_fp = fp[i]
        s_fn = fn[i]

        if (s_tp + s_fp) != 0 and (s_tp + s_fn) != 0 and (s_tn + s_fp) != 0 and (s_tn + s_fn) != 0:
            MCC = (s_tp * s_tn - s_fp * s_fn) / math.sqrt((s_tp + s_fp) * (s_tp + s_fn) * (s_tn + s_fp) * (s_tn + s_fn))
        else:
            MCC = 0

        if (s_tp + s_fn) != 0 and (s_fp + s_tn) != 0:
            GM = math.sqrt((s_tp / (s_tp + s_fn)) * (s_tn / (s_fp + s_tn)))
        else:
            GM = 0
        MCC_list.append(str(MCC))
        GM_list.append(str(GM))

    ##write out results
    ##all the evaluaton scores are array
    store_final_line_list = []
    ##generate the first line
    first_line = 'Cell_type' + '\t' + 'AC' + '\t' + 'SE' + '\t' + 'PR' + '\t' + \
                 'SP' + '\t' + 'MCC' + '\t' + 'GM' + '\t' + 'MAP'
    store_final_line_list.append(first_line)

    for i in range(len(list(id_nm_dic.keys()))):
        final_line = id_nm_dic[str(i)] + '\t' + str(AC[i]) + '\t' + str(SE[i]) + '\t' + \
                     str(PR[i]) + '\t' + str(SP[i]) + '\t' + str(MCC_list[i]) + '\t' + \
                     str(GM_list[i]) + '\t' + str(cell_type_map[0][i])
        store_final_line_list.append(final_line)

    with open(input_opt_dir + '/opt_' + model_name + '_' + test_type + '_class_evaluation_score.txt', 'w+') as opt:
        for eachline in store_final_line_list:
            opt.write(eachline + '\n')


def train_evaluation (input_train_file,input_test_file,input_meta_train_file,input_meta_test_file,
                      input_test_indep_file,input_meta_test_indep_file,input_opt_dir):

    ##run script
    ##load import file
    features_train = read_csv(input_train_file, index_col = 0).values.T
    features_test =  read_csv(input_test_file, index_col = 0).values.T
    meta_train = read_csv(input_meta_train_file,header = 0, index_col = 0)
    meta_test = read_csv(input_meta_test_file,header = 0, index_col = 0)

    label_names = meta_train.loc[:,"cell_type"].unique()
    names_to_id = { key:val for val,key in enumerate(label_names)}
    labels_train = meta_train.loc[:,"cell_type"].replace(names_to_id).values
    labels_test = meta_test.loc[:,"cell_type"].replace(names_to_id).values

    ##initiate a function to store single cell id and its real name
    ##change the order of id and name
    id_nm_dic = {}
    for eachnm in names_to_id:
        id_nm_dic[str(names_to_id[eachnm])] = eachnm

    #########
    ##For svm
    clf_svm = svm.SVC(kernel='linear',probability=True)
    clf_svm.fit(features_train, labels_train)
    ##save svm model
    pkl_filename = input_opt_dir + "/svm_model.pkl"
    with open(pkl_filename, 'wb') as file:
        pickle.dump(clf_svm, file)

    ##for the validation data
    clfs = [clf_svm]
    labels_pred = []
    prob_pred = []
    for clf in clfs:
        prob_pred_all = clf.predict_proba(features_test)
        labels_pred.append(np.argsort(-prob_pred_all,axis = 1)[:,0])
        prob_pred.append(np.array([prob_pred_all[i,labels_pred[-1][i]] for i in range(labels_pred[-1].shape[0])]))
    ##evaluation
    evaluation (labels_pred,prob_pred,labels_test,label_names,'validate',id_nm_dic,'svm',input_opt_dir)

    ##for the independent data
    features_indep_test =  read_csv(input_test_indep_file, index_col = 0).values.T
    meta_indep_test = read_csv(input_meta_test_indep_file,header = 0, index_col = 0)
    labels_indep_test = meta_indep_test.loc[:,"cell_type"].replace(names_to_id).values
    labels_indep_pred = []
    prob_indep_pred = []
    for clf in clfs:
        prob_pred_all = clf.predict_proba(features_indep_test)
        labels_indep_pred.append(np.argsort(-prob_pred_all,axis = 1)[:,0])
        prob_indep_pred.append(np.array([prob_pred_all[i,labels_indep_pred[-1][i]] for i in range(labels_indep_pred[-1].shape[0])]))
    ##evaluation
    evaluation (labels_indep_pred,prob_indep_pred,labels_indep_test,label_names,'independent',id_nm_dic,'svm',input_opt_dir)

    ########
    ##For RF
    clf_rf = RandomForestClassifier(n_estimators = 500, n_jobs = 42).fit(features_train, labels_train)
    ##save rf model
    pkl_filename = input_opt_dir + "/rf_model.pkl"
    with open(pkl_filename, 'wb') as file:
        pickle.dump(clf_rf, file)

    ##for the validation data
    clfs = [clf_rf]
    labels_pred = []
    prob_pred = []
    for clf in clfs:
        prob_pred_all = clf.predict_proba(features_test)
        labels_pred.append(np.argsort(-prob_pred_all,axis = 1)[:,0])
        prob_pred.append(np.array([prob_pred_all[i,labels_pred[-1][i]] for i in range(labels_pred[-1].shape[0])]))
    ##evaluation
    evaluation (labels_pred,prob_pred,labels_test,label_names,'validate',id_nm_dic,'rf',input_opt_dir)

    ##for the independent data
    features_indep_test =  read_csv(input_test_indep_file, index_col = 0).values.T
    meta_indep_test = read_csv(input_meta_test_indep_file,header = 0, index_col = 0)
    labels_indep_test = meta_indep_test.loc[:,"cell_type"].replace(names_to_id).values
    labels_indep_pred = []
    prob_indep_pred = []
    for clf in clfs:
        prob_pred_all = clf.predict_proba(features_indep_test)
        labels_indep_pred.append(np.argsort(-prob_pred_all,axis = 1)[:,0])
        prob_indep_pred.append(np.array([prob_pred_all[i,labels_indep_pred[-1][i]] for i in range(labels_indep_pred[-1].shape[0])]))
    ##evaluation
    evaluation (labels_indep_pred,prob_indep_pred,labels_indep_test,label_names,'independent',id_nm_dic,'rf',input_opt_dir)