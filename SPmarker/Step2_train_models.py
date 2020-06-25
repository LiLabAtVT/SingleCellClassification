#!/usr/bin/env python

##Step 2 prepare and train models

##BUILT-IN MODULES
import os
import argparse
import sys
import subprocess
import glob
import re

from utils import S2_0_select_feature as S2_0_select_feat
from utils import S2_1_generate_dataset_from_ICI_res as S2_1_generate_data
from utils import S2_2_split_dataset_to_train_test as S2_2_split_data
from utils import S2_3_split_dataset_cross_val as S2_3_split_cross_val
from utils import S2_4_train_model_evaluation as S2_4_train_model_eval
from utils import S2_5_collect_eval_res as S2_5_collect_eval_res

def get_parsed_args():

    parser = argparse.ArgumentParser(description="SPmarker train models")

    ##require files
    parser.add_argument("-d", dest='working_dir', default="./", help="Working directory to store intermediate files of "
                                                                     "each step. Default: ./ ")

    parser.add_argument("-o", dest='output_dir', default="./", help="Output directory to store the output files."
                                                                    "Default: ./ ")

    ##optional file
    ##choice 1: provide the ICI file and exp files
    parser.add_argument("-ICI_fl", dest="ICI_file",help="Provide the ICI file with csv format."
                                                       "This ICI file is generated from Step1.1_assign_cell_type_using_ICI.R")

    parser.add_argument("-exp_fl",dest="expression_data", help="Provide the cell expression data with csv format."
                                                              "colname is cell name and rowname is feature name.")

    ##choice 2: directly provide the exp and meta files
    ##it also needs the expression data
    parser.add_argument("-meta_fl",dest="meta_data",help="Provide the cell meat with csv format."
                                                        "rowname is cell name and "
                                                        "colname contain cell type")

    ##optinal feature select
    ##if users do not provide this file we will not conduct this step
    parser.add_argument('-feat_fl',dest="feature_file",help="Provide the features that will be kept in the expression file."
                                                           "If users do not provide the argument, we will use all the features.")


    ##optional parameter setting
    ##parameters for the choice 1: S2_1_generate_data
    parser.add_argument("-thr_ICI",dest="threshold_ICI",help="Provide a threshold to remove cells with low probability."
                                                             "Default: 0.5")

    ##parameters for the S2_2_split_data
    parser.add_argument("-div", dest='divide_pro', help="Proportion of training and testing dataset."
                                                 "Default: 0.1."
                                                 "The 0.1 indicates 10% of the data are the testing data"
                                                 "and 90% of the data are the training data.")

    parser.add_argument("-max_num", dest='max_cell_number', help="Provide the max cell number."
                                                                 "If cell type with the cell number over the max cell number,"
                                                                 "we will change the number of the cell type to the max number that users set."
                                                                 "Default: 5000")

    parser.add_argument("-min_num", dest='min_cell_number', help="Provide the minimum cell number."
                                                                 "If cell type with the cell number below the minimum cell number,"
                                                                 "this cell type will be removed out."
                                                                 "Default: 100")

    parser.add_argument("-fd_num", dest='fold_number', help="Provide cross validation folds."
                                                            "Default: 5")

    parser.add_argument("-eval", dest="evaluation_score", help="Provide the evaluation score that will be used to select the best random forest model to conduct the SHAP analysis."
                                                               "Default: MCC")


    ##parse of parameters
    args = parser.parse_args()
    return args



def main(argv=None):
    if argv is None:
        argv = sys.argv
    args = get_parsed_args()


    #######################################
    ##check the required software and files
    ##for the input files
    if args.ICI_file is not None:

        try:
            file = open(args.ICI_file ,'r')
        except IOError:
            print('There was an error opening the ICI_file!')
            return

        if args.expression_data is None:
            print('Cannot find expression data, please provide it')
            return
        else:
            try:
                file = open(args.expression_data, 'r')  ##check if the file is not the right file
            except IOError:
                print('There was an error opening the expression data!')
                return

        if args.meta_data is not None:
            print('Do not provide the meta data since we already provide the ICI file')
            return

    else:

        if args.expression_data is None:
            print('Cannot find expression data, please provide it')
            return
        else:
            try:
                file = open(args.expression_data, 'r')  ##check if the file is not the right file
            except IOError:
                print('There was an error opening the expression data!')
                return

        if args.meta_data is None:
            print('Cannot find meta data, please provide it')
            return
        else:
            try:
                file = open(args.meta_data, 'r')  ##check if the file is not the right file
            except IOError:
                print('There was an error opening the meta data!')
                return

    if args.feature_file is not None:
        try:
            file = open(args.feature_file ,'r')
        except IOError:
            print('There was an error opening the feature_file!')
            return

    ##for the parameters
    ##create the meme number
    if args.threshold_ICI is not None:
        threshold_ICI = args.threshold_ICI
    else:
        threshold_ICI = '0.5'

    if args.divide_pro is not None:
        divide_pro = args.divide_pro
    else:
        divide_pro = '0.1'

    if args.max_cell_number is not None:
        max_cell_number = args.max_cell_number
    else:
        max_cell_number = '5000'

    if args.min_cell_number is not None:
        min_cell_number = args.min_cell_number
    else:
        min_cell_number = '100'

    if args.fold_number is not None:
        fold_number = args.fold_number
    else:
        fold_number = '5'

    if args.evaluation_score is not None:
        evaluation_score = args.evaluation_score
    else:
        evaluation_score = 'MCC'

    ###########################################
    ##create the working and output directories
    working_dir = args.working_dir
    if not working_dir.endswith('/'):
        working_dir = working_dir + '/'
    else:
        working_dir = working_dir

    output_dir = args.output_dir
    if not output_dir.endswith('/'):
        output_dir = output_dir + '/'
    else:
        output_dir = output_dir

    #################
    ##Run the process

    ########
    ##Step 0 prepare the input exp
    if args.feature_file is not None:

        ipt_feature_file = args.feature_file
        ipt_expression_data = args.expression_data
        ##create a dir under the working_dir
        S0_feature_selection_dir = working_dir + '/S0_feature_selection_dir'
        if not os.path.exists(S0_feature_selection_dir):
            os.makedirs(S0_feature_selection_dir)

        ##write results
        exp_line_list = S2_0_select_feat.extract_marker_gene(ipt_expression_data, ipt_feature_file)
        with open(S0_feature_selection_dir + '/opt_select_feat_exp.csv', 'w+') as opt:
            for eachline in exp_line_list:
                opt.write(eachline + '\n')

        new_exp_fl_path = S0_feature_selection_dir + '/opt_select_feat_exp.csv'

    else:
        new_exp_fl_path = args.expression_data

    ########
    ##Step 1 generate meta from ICI or not
    print('Step 1 Generate meta')
    store_cell_nm_dic = {} ##cell name is the key and the value is the cell number
    if args.ICI_file is not None:

        ipt_ICI_file = args.ICI_file
        ##create a dir under the working_dir
        S1_construct_meta_from_ICI_dir = working_dir + '/S1_construct_meta_from_ICI_dir'
        if not os.path.exists(S1_construct_meta_from_ICI_dir):
            os.makedirs(S1_construct_meta_from_ICI_dir)

        S2_1_generate_data.generate_dataset_ICI(ipt_ICI_file, S1_construct_meta_from_ICI_dir)

        store_final_line, summary_cell_type_before_filter_dic, \
        summary_cell_type_after_filter_dic, \
        store_filter_single_cell_id_dic = S2_1_generate_data.filter_out_pvalue(S1_construct_meta_from_ICI_dir + '/raw_meta.csv',
                                                                               threshold_ICI)

        with open(S1_construct_meta_from_ICI_dir + '/opt_filter_cell_meta.csv', 'w+') as opt:
            for eachline in store_final_line:
                opt.write(eachline + '\n')

        new_meta_fl_path = S1_construct_meta_from_ICI_dir + '/opt_filter_cell_meta.csv'


        with open(S1_construct_meta_from_ICI_dir + '/summary_cell_type_before_filter.txt', 'w+') as opt:
            for eachnm in summary_cell_type_before_filter_dic:
                opt.write(eachnm + '\t' + str(summary_cell_type_before_filter_dic[eachnm]) + '\n')

        with open(S1_construct_meta_from_ICI_dir + '/summary_cell_type_after_filter.txt', 'w+') as opt:
            for eachnm in summary_cell_type_after_filter_dic:
                opt.write(eachnm + '\t' + str(summary_cell_type_after_filter_dic[eachnm]) + '\n')

        store_cell_nm_dic = summary_cell_type_after_filter_dic

        with open(S1_construct_meta_from_ICI_dir + '/filter_single_cell_id.txt', 'w+') as opt:
            for eachid in store_filter_single_cell_id_dic:
                opt.write(eachid + '\n')

        exp_filter_id_dt = S2_1_generate_data.store_remain_id_in_exp_data(new_exp_fl_path, store_filter_single_cell_id_dic)
        exp_filter_id_dt.to_csv(S1_construct_meta_from_ICI_dir + '/opt_filter_cell_exp.csv', index=True)

        new_filter_exp_fl_path = S1_construct_meta_from_ICI_dir + '/opt_filter_cell_exp.csv'

    else:
        new_filter_exp_fl_path = new_exp_fl_path
        new_meta_fl_path = args.meta_data

        ##generate a script to calculate the number of cell for each cells according to this user provided meta data
        with open (new_meta_fl_path,'r') as ipt:
            for eachline in ipt:
                eachline = eachline.strip('\n')
                col = eachline.strip().split()
                if col[1] in store_cell_nm_dic:
                    store_cell_nm_dic[col[1]] += 1
                else:
                    store_cell_nm_dic[col[1]] = 1

    ########
    ##Step 2 split the data
    print('Step 2 Split data')
    S2_split_dataset_dir = working_dir + '/S2_split_dataset_dir'
    if not os.path.exists(S2_split_dataset_dir):
        os.makedirs(S2_split_dataset_dir)

    ##filter out cell
    input_fl_out_fam_list = [] ##input_fl_out_fam_list = ['Phloem', 'Late.PPP', 'Pericycle', 'LRM', 'Late.XPP']
    cell_type_dic = {} ##cell_type_dic = {'Cortext': 5000, 'Atrichoblast': 5000}
    for eachcell_type in store_cell_nm_dic:
        cell_type_num = store_cell_nm_dic[eachcell_type]

        if int(cell_type_num) < int(min_cell_number):
            input_fl_out_fam_list.append(eachcell_type)
        else:
            if int(cell_type_num) > int(max_cell_number):
                cell_type_dic[eachcell_type] = int(max_cell_number)

    ##run the split process
    S2_2_split_data.split_train_test(new_filter_exp_fl_path, new_meta_fl_path, divide_pro,
                                     S2_split_dataset_dir, input_fl_out_fam_list,cell_type_dic)

    ##extract family number
    ##copy this file to the outputdir
    cmd = 'cp ' + S2_split_dataset_dir + '/opt_fam_nm_num.txt ' + output_dir + '/opt_train_validation_cell_type_number.txt'
    subprocess.call(cmd,shell=True)

    ########
    ##Step 3 split the data to five fold cross validations
    print('Step 3 Split train data used in cross validation')
    S3_split_cross_val_dir = working_dir + '/S3_split_cross_val_dir'
    if not os.path.exists(S3_split_cross_val_dir):
        os.makedirs(S3_split_cross_val_dir)

    ##generate working and output directories
    S3_split_cross_val_w_dir = S3_split_cross_val_dir + '/working_dir'
    if not os.path.exists(S3_split_cross_val_w_dir):
        os.makedirs(S3_split_cross_val_w_dir)

    S3_split_cross_val_o_dir = S3_split_cross_val_dir + '/output_dir'
    if not os.path.exists(S3_split_cross_val_o_dir):
        os.makedirs(S3_split_cross_val_o_dir)

    ##use the generated opt_exp_train.csv to conduct the cross validation
    input_exp_fl = S2_split_dataset_dir + '/opt_exp_train.csv'
    input_meta_fl = S2_split_dataset_dir + '/opt_meta_train.csv'

    input_inde_exp_fl = S2_split_dataset_dir + '/opt_exp_test.csv'
    input_inde_meta_fl = S2_split_dataset_dir + '/opt_meta_test.csv'

    S2_3_split_cross_val.split_data_five_fold_cross_val(input_meta_fl, input_exp_fl, S3_split_cross_val_w_dir, S3_split_cross_val_o_dir,int(fold_number))

    ##copy indep testing data to output dir. the data will be used for detecting shap marker
    cmd = 'cp ' + input_inde_exp_fl + ' ' + output_dir + '/opt_exp_indep_test.csv'
    subprocess.call(cmd,shell=True)

    cmd = 'cp ' + input_inde_meta_fl + ' ' + output_dir + '/opt_meta_indep_test.csv'
    subprocess.call(cmd,shell=True)

    ##cp the independent exp and meta fl to the output dir from the the split cross validation dir
    cross_vali_dir_list = glob.glob(S3_split_cross_val_o_dir + '/*')
    for eachcross_dir in cross_vali_dir_list:
        ##eachcross is a b c d e dir
        cmd = 'cp ' + input_inde_exp_fl + ' ' + eachcross_dir + '/opt_exp_indep_test.csv'
        subprocess.call(cmd,shell=True)

        cmd = 'cp ' + input_inde_meta_fl + ' ' + eachcross_dir + '/opt_meta_indep_test.csv'
        subprocess.call(cmd,shell=True)

    ##now the S3_split_cross_val_o_dir contains all the files that the training needs

    ########
    ##Step 4 train the models
    print('Step 4 Model training and evaluation')
    S4_train_model_eval_o_dir = working_dir + '/S4_train_model_eval_o_dir'
    if not os.path.exists(S4_train_model_eval_o_dir):
        os.makedirs(S4_train_model_eval_o_dir)

    ipt_dt_list = glob.glob(S3_split_cross_val_o_dir + '/*')
    for eachipt_dir in ipt_dt_list:
        mt = re.match('.+/(.+)',eachipt_dir)
        rep_nm = mt.group(1) ##rep_nm is a, b, c, d, e

        ##generate output dir
        rep_output_dir = S4_train_model_eval_o_dir + '/' + rep_nm
        if not os.path.exists(rep_output_dir):
            os.makedirs(rep_output_dir)

        ipt_exp_train_fl = eachipt_dir + '/opt_exp_train.csv'
        ipt_meta_train_fl = eachipt_dir + '/opt_meta_train.csv'
        ipt_exp_test_fl = eachipt_dir + '/opt_exp_test.csv'
        ipt_meta_test_fl = eachipt_dir + '/opt_meta_test.csv'

        ipt_inde_exp_test_fl = eachipt_dir + '/opt_exp_indep_test.csv'
        ipt_inde_meta_test_fl = eachipt_dir + '/opt_meta_indep_test.csv'

        ##run the training process
        ##run script
        ##train both models since we also generate the svm and rf or shap markers at the same time
        S2_4_train_model_eval.train_evaluation(ipt_exp_train_fl, ipt_exp_test_fl, ipt_meta_train_fl, ipt_meta_test_fl,
                                               ipt_inde_exp_test_fl, ipt_inde_meta_test_fl, rep_output_dir)

    ########
    ##Step 5 collect the results
    ##collect the results
    print('Step 5 Collect model performance')
    S5_collect_res_o_dir = working_dir + '/S5_collect_res_o_dir'
    if not os.path.exists(S5_collect_res_o_dir):
        os.makedirs(S5_collect_res_o_dir)

    method_list = ['svm', 'rf']
    S2_5_collect_eval_res.extract_output (S4_train_model_eval_o_dir,S5_collect_res_o_dir,method_list)

    ########
    ##Step 6 model select for the rf models
    print('Step 6 select the best rf model and collect all svm models')
    ##the output contains two dirs: rf and svm
    ##rf is used to compare to select the best model for detecting the SHAP markers
    ##svm model is used to extract feature importance to generate SVM markers
    ##we need to compare performance of each model using the testing evaluation
    S2_5_collect_eval_res.select_best_model(S4_train_model_eval_o_dir, output_dir, S5_collect_res_o_dir, evaluation_score)

    ##we need to cp all the svm models to the output_dir
    ##generate a output in the final output dir
    opt_store_svm_models_dir = working_dir + '/opt_store_svm_models_dir'
    if not os.path.exists(opt_store_svm_models_dir):
        os.makedirs(opt_store_svm_models_dir)

    opt_rep_dir_list = glob.glob(S4_train_model_eval_o_dir + '/*')
    for eachrep_dir in opt_rep_dir_list:

        mt = re.match('.+/(.+)',eachrep_dir)
        rep_nm = mt.group(1)

        opt_store_svm_model_rep_nm_dir = opt_store_svm_models_dir + '/' + rep_nm
        if not os.path.exists(opt_store_svm_model_rep_nm_dir):
            os.makedirs(opt_store_svm_model_rep_nm_dir)

        cmd = 'cp ' + eachrep_dir + '/svm_model.pkl ' + opt_store_svm_model_rep_nm_dir
        subprocess.call(cmd,shell=True)

    ##the opt_store_svm_models_dir will be used for the SVM marker detection

if __name__ == "__main__":
    main()
















