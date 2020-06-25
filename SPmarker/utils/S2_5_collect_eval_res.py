#!/usr/bin/env python

##updation 060320 add a function to select the best rf model
##copy this model to the final output dir
##this script will collect data from var02 svm and rf
##this script will collect output in the arc to be consistent with input in the evaluation_cross_val_041520 dir
##because we generate evaluation by different ways

import glob
import os
import re
import subprocess
import pandas as pd

def extract_output (input_opt_dir,output_dir,method_list):

    for eachmethod_nm in method_list:

        opt_method_dir = output_dir + '/' + eachmethod_nm
        if not os.path.exists(opt_method_dir):
            os.makedirs(opt_method_dir)

        ##generate two output dir testing and validation
        opt_testing_dir = opt_method_dir + '/testing'
        if not os.path.exists(opt_testing_dir):
            os.makedirs(opt_testing_dir)

        opt_validation_dir = opt_method_dir + '/validation'
        if not os.path.exists(opt_validation_dir):
            os.makedirs(opt_validation_dir)

        rep_dir_list = glob.glob(input_opt_dir + '/*')
        for eachrep_dir in rep_dir_list:
            mt = re.match('.+/(.+)',eachrep_dir)
            rep_nm = mt.group(1) ##rep_nm is a b c d e

            fl_list = glob.glob(eachrep_dir + '/*class_evaluation_score.txt')
            for eachfl in fl_list:

                mt = re.match('.+/(.+)',eachfl)
                fl_nm = mt.group(1)

                mt = re.match('opt_(.+)_(.+)_class_evaluation_score\.txt',fl_nm)
                method_nm = mt.group(1)
                test_type = mt.group(2)

                if method_nm == eachmethod_nm:
                    ##cp the fl to the output file
                    ##consider two situations
                    if test_type == 'independent':
                        cmd = 'cp ' + eachfl + ' ' + opt_testing_dir + '/opt_' + rep_nm + '_class_evaluation_score.txt'
                        print(cmd)
                        subprocess.call(cmd,shell=True)
                    if test_type == 'validate':
                        cmd = 'cp ' + eachfl + ' ' + opt_validation_dir + '/opt_' + rep_nm + '_class_evaluation_score.txt'
                        print(cmd)
                        subprocess.call(cmd,shell=True)


def select_best_model (input_opt_store_model_dir,final_output_dir,collect_opt_dir,eval_score_nm):

    store_rep_nm_dic = {} ##key is the rep_nm and value is the eval score
    opt_eval_score_fl_list = glob.glob(collect_opt_dir + '/rf/testing/*')
    for eacheval_score_fl in opt_eval_score_fl_list:

        mt = re.match('.+/(.+)',eacheval_score_fl)
        fl_nm = mt.group(1)

        mt = re.match('opt_(.+)_class_evaluation_score\.txt',fl_nm)
        rep_nm = mt.group(1)

        dt = pd.read_csv(eacheval_score_fl,sep='\t')
        ##updation 060620 change the mean_eval_score to the sum_eval_score since MCC has negative value
        sum_eval_score = dt[eval_score_nm].sum(axis=0)

        store_rep_nm_dic[rep_nm] = float(sum_eval_score)

    largest_rep_nm = max(store_rep_nm_dic, key=store_rep_nm_dic.get)

    ##cp the best model the final_output_dir
    cmd = 'cp ' +  input_opt_store_model_dir + '/' + largest_rep_nm + '/rf_model.pkl ' + final_output_dir
    subprocess.call(cmd,shell=True)