#!/usr/bin/env python

##set to the remain number
##updation 041120 add all meta output
##updation 040120
# this script will randomly remove the 50% selected single cell

##updation 012720 decrease the number of category with large number of single number
##first we sort the value of the category and extract the first xx part of data

##this script is to divide the dataset into test and train dataset
## 5% 95%
##divide each group into 5% and 95%
##generate 5% random id and 95% remained id

import os
import pandas as pd
import random
import numpy as np


def split_train_test(input_dataset_exp_fl, input_meta_fl, input_thr_par, input_opt_dir, input_fl_out_fam_list,
                     cell_type_dic):
    ##generate a temp dir to store the temp meta data for the cell type whose size will be decreased
    opt_temp_store_meta_dir = input_opt_dir + '/opt_temp_store_meta_dir'
    if not os.path.exists(opt_temp_store_meta_dir):
        os.makedirs(opt_temp_store_meta_dir)

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

    ##value is the list of sg cell name
    total_sg_cell_list = []

    train_sg_cell_list = []
    test_sg_cell_list = []
    for eachfamnm in fam_nm_dic:

        if eachfamnm not in input_fl_out_fam_list:  ##it does not work

            total_fam_count = 0
            sg_cell_list = []
            ##updaiton 012720
            ##check if the eachfamnm is in the cell_type_dic
            ##if this generate a temp dir to sort the data from this cell type
            if eachfamnm in cell_type_dic:

                raw_meta_fl_dt = pd.read_csv(input_meta_fl, header=0, index_col=0)
                ##select the specific fam name
                famn_dt = raw_meta_fl_dt.loc[raw_meta_fl_dt['cell_type'] == eachfamnm]

                ##updation 040120 shuffle the dataframe
                shf_famn_dt = famn_dt.iloc[np.random.permutation(len(famn_dt))]

                ##sort the value in the famn_dt
                # sort_famn_dt = famn_dt.sort_values(by='prob',ascending=False)

                ##save the dt
                shf_famn_dt.to_csv(opt_temp_store_meta_dir + '/' + eachfamnm + '.csv', index=True)

                ##decrease the size of specific family
                remain_cell_num = cell_type_dic[eachfamnm]
                #remain_cell_num = de_por * int(shf_famn_dt.shape[0])

                de_shf_famn_dt = shf_famn_dt.head(int(remain_cell_num))
                ##save to the dir
                de_shf_famn_dt.to_csv(opt_temp_store_meta_dir + '/' + eachfamnm + '_final.csv', index=True)

                ##read the new meta fl
                count = 0
                with open(opt_temp_store_meta_dir + '/' + eachfamnm + '_final.csv', 'r') as ipt:
                    for eachline in ipt:
                        count += 1
                        eachline = eachline.strip('\n')
                        if count != 1:
                            col = eachline.strip().split(',')
                            if eachfamnm == col[1]:
                                total_fam_count += 1
                                sg_cell_nm = col[0]
                                sg_cell_list.append(sg_cell_nm)

            else:
                ##this type will not be analyzed
                count = 0
                with open(input_meta_fl, 'r') as ipt:
                    for eachline in ipt:
                        count += 1
                        eachline = eachline.strip('\n')
                        if count != 1:
                            col = eachline.strip().split(',')
                            if eachfamnm == col[1]:
                                total_fam_count += 1
                                sg_cell_nm = col[0]
                                sg_cell_list.append(sg_cell_nm)

            ##shuffle the list
            random.shuffle(sg_cell_list)
            fam_count = 0

            for eachid in sg_cell_list:
                fam_count += 1
                if fam_count <= int(total_fam_count * float(input_thr_par)):
                    test_sg_cell_list.append(eachid)
                else:
                    train_sg_cell_list.append(eachid)

                total_sg_cell_list.append(eachid)

    ##target on the meta and exp data
    ##for the exp data
    raw_exp_fl_dt = pd.read_csv(input_dataset_exp_fl, header=0, index_col=0)
    exp_test_id_dt = raw_exp_fl_dt[test_sg_cell_list]
    exp_test_id_dt.to_csv(input_opt_dir + '/opt_exp_test.csv', index=True)
    exp_train_id_dt = raw_exp_fl_dt[train_sg_cell_list]
    exp_train_id_dt.to_csv(input_opt_dir + '/opt_exp_train.csv', index=True)

    ##updaton 040120 set a total sg cell list to do the feature selection
    exp_id_dt = raw_exp_fl_dt[total_sg_cell_list]
    exp_id_dt.to_csv(input_opt_dir + '/opt_exp_all.csv', index=True)

    ##for the meta data
    raw_meta_fl_dt = pd.read_csv(input_meta_fl, header=0, index_col=0)
    meta_test_id_dt = raw_meta_fl_dt.loc[test_sg_cell_list]
    meta_test_id_dt.to_csv(input_opt_dir + '/opt_meta_test.csv', index=True)
    meta_train_id_dt = raw_meta_fl_dt.loc[train_sg_cell_list]
    meta_train_id_dt.to_csv(input_opt_dir + '/opt_meta_train.csv', index=True)

    ##updation 041120 add opt_meta_all.csv
    meta_id_dt = raw_meta_fl_dt.loc[total_sg_cell_list]
    meta_id_dt.to_csv(input_opt_dir + '/opt_meta_all.csv', index=True)

    ##check the number of meta data
    store_meta_test_nm_dic = {}
    store_meta_train_nm_dic = {}
    with open(input_opt_dir + '/opt_meta_test.csv', 'r') as ipt:
        for eachline in ipt:
            count += 1
            eachline = eachline.strip('\n')
            if count != 1:
                col = eachline.strip().split(',')
                if col[1] in store_meta_test_nm_dic:
                    store_meta_test_nm_dic[col[1]] += 1
                else:
                    store_meta_test_nm_dic[col[1]] = 1

    with open(input_opt_dir + '/opt_meta_train.csv', 'r') as ipt:
        for eachline in ipt:
            count += 1
            eachline = eachline.strip('\n')
            if count != 1:
                col = eachline.strip().split(',')
                if col[1] in store_meta_train_nm_dic:
                    store_meta_train_nm_dic[col[1]] += 1
                else:
                    store_meta_train_nm_dic[col[1]] = 1

    ##change the col name of the opt_fam_nm_num.txt
    with open(input_opt_dir + '/opt_fam_nm_num.txt', 'w+') as opt:
        for nm in store_meta_test_nm_dic:
            if nm == 'cell_type':
                opt.write('Cell_type' + '\t' + 'Validation' + '\t' + 'Training' + '\n')
            else:
                opt.write(nm + '\t' + str(store_meta_test_nm_dic[nm]) + '\t' + \
                      str(store_meta_train_nm_dic[nm]) + '\n')

