#!/usr/bin/env python

##this script will generate five datasets using five cross validation for each familiy
##if we need to have other number of cross validation we need to change the five to ten


import os
import pandas as pd
import numpy as np
from itertools import islice


def split_data_five_fold_cross_val (input_meta_fl,input_exp_fl,input_working_dir,input_output_dir,fold_num):

    ##store sg cell name in each family in a dic
    ##key is the cell name
    fam_nm_dic = {}
    count = 0
    with open(input_meta_fl, 'r') as ipt:
        for eachline in ipt:
            count += 1
            eachline = eachline.strip('\n')
            if count != 1:
                col = eachline.strip().split(',')
                fam_nm = col[1]
                fam_nm_dic[fam_nm] = 1

    ##mk a dir to store the replicates from each single cell type
    divide_family_output_dir = input_working_dir + '/divide_family_output'
    if not os.path.exists(divide_family_output_dir):
        os.makedirs(divide_family_output_dir)

    #############################
    ##store all the path together
    store_famnm_split_cnt_dic = {}  ##key is the famnm value is another dic whose key is str(count) and value is the path of csv
    for eachfamnm in fam_nm_dic:

        ##create a fam dir
        fam_divide_dir = input_working_dir + '/' + eachfamnm + '_dir'
        if not os.path.exists(fam_divide_dir):
            os.makedirs(fam_divide_dir)


        ##work in the fam_divide_dir
        raw_meta_fl_dt = pd.read_csv(input_meta_fl, header=0, index_col=0)
        ##select the specific fam name
        famn_dt = raw_meta_fl_dt.loc[raw_meta_fl_dt['cell_type'] == eachfamnm]
        ##generate shuffle dt
        shf_famn_dt = famn_dt.iloc[np.random.permutation(len(famn_dt))]
        ##save the dt
        shf_famn_dt.to_csv(fam_divide_dir + '/shuffle_' + eachfamnm + '.csv', index=True)

        ##collect the index of shf_famn_dt to one list
        index_list = list(shf_famn_dt.index.values)

        num, div = len(index_list), fold_num ##five fold cross validation
        length_to_split = [num // div + (1 if x < num % div else 0) for x in range(div)]

        Inputt = iter(index_list)
        split_index_list = [list(islice(Inputt, elem)) for elem in length_to_split]
        ##this split_index_list contains three lists of each contains

        ##split the dataframes into five dataframes
        split_count = 0
        ##store the split dataframe path into a dic
        store_split_path_dic = {}
        for eachindex_list in split_index_list:
            split_shf_famn_dt = shf_famn_dt.loc[eachindex_list]
            split_count += 1
            ##write to csv file
            split_shf_famn_dt.to_csv(fam_divide_dir + '/split_' + str(split_count) + '_' + eachfamnm + '.csv',index=True)
            store_split_path_dic[str(split_count)] = fam_divide_dir + '/split_' + str(split_count) + '_' + eachfamnm + '.csv'

        ##store the store_split_path_dic to the store_famnm_split_cnt_dic
        store_famnm_split_cnt_dic[eachfamnm] = store_split_path_dic

    ########################
    ##generate five datasets
    letter_list = ['a', 'b', 'c', 'd', 'e','f','g','h','i','j']
    let_count = 0
    split_count = 0
    for each_let_nm in letter_list:
        let_count += 1

        ##do not exceed the fold number
        if let_count <= fold_num:

            split_count += 1

            ##generate test dt
            test_dt_frames = []
            train_dt_frames = []
            for eachfamnm in store_famnm_split_cnt_dic:
                store_split_path_dic = store_famnm_split_cnt_dic[eachfamnm]
                for eachsplit_str in store_split_path_dic:
                    if eachsplit_str == str(split_count):
                        fam_split_path = store_split_path_dic[eachsplit_str]
                        fam_split_dt = pd.read_csv(fam_split_path, header=0, index_col=0)
                        ##combine all the fam split path together
                        ##this is for the test
                        test_dt_frames.append(fam_split_dt)

                    ##it means b c d e if the previous one is a
                    else:
                        fam_split_path = store_split_path_dic[eachsplit_str]
                        fam_split_dt = pd.read_csv(fam_split_path, header=0, index_col=0)
                        train_dt_frames.append(fam_split_dt)

            combine_test_frame_dt = pd.concat(test_dt_frames)
            combine_train_frame_dt = pd.concat(train_dt_frames)
            ##write out frame dataframe
            ##mk a dir to store the data from combining family
            combine_family_output_dir = input_output_dir + '/' + each_let_nm
            if not os.path.exists(combine_family_output_dir):
                os.makedirs(combine_family_output_dir)

            combine_test_frame_dt.to_csv(combine_family_output_dir + '/opt_meta_test.csv', index=True)
            combine_train_frame_dt.to_csv(combine_family_output_dir + '/opt_meta_train.csv',index=True)

            ##generate expression
            ##extract index list
            ##for the testing
            exp_fl_dt = pd.read_csv(input_exp_fl, header=0, index_col=0)
            combine_test_dt_index_list = list(combine_test_frame_dt.index.values)
            exp_test_dt = exp_fl_dt[combine_test_dt_index_list]
            exp_test_dt.to_csv(combine_family_output_dir + '/opt_exp_test.csv', index=True)

            ##for the training
            combine_train_dt_index_list = list(combine_train_frame_dt.index.values)
            exp_train_dt = exp_fl_dt[combine_train_dt_index_list]
            exp_train_dt.to_csv(combine_family_output_dir + '/opt_exp_train.csv', index=True)



















