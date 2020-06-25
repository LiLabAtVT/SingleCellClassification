#!/usr/bin/env python

import pandas as pd

def generate_dataset_ICI (input_ICI_file,input_opt_dir):

    ICI_file_df = pd.read_csv(input_ICI_file,header = 0, index_col = 0)
    ICI_file_df['Max_col_nm'] = ICI_file_df.idxmax(axis=1)  ##find the column with the greatest value on each row
    ICI_file_df['max_value'] = ICI_file_df.max(axis=1)  ##Get max value from row of a dataframe in python [duplicate]
    raw_meta_df = ICI_file_df[['Max_col_nm','max_value']]
    raw_meta_df = raw_meta_df.rename(columns={"Max_col_nm": "cell_type", "max_value": "prob"})
    raw_meta_df.to_csv(input_opt_dir + '/raw_meta.csv',index=True)

##define a function to filter out prob that is below to thr pvalue
def filter_out_pvalue (raw_meta_file,input_thr_value):

    store_final_line = []
    summary_cell_type_before_filter_dic = {}
    summary_cell_type_after_filter_dic = {}
    store_filter_single_cell_id_dic = {}
    count = 0
    with open (raw_meta_file,'r') as ipt:
        for eachline in ipt:
            count += 1
            eachline = eachline.strip('\n')
            if count != 1:
                col = eachline.strip().split(',')
                if col[1] in summary_cell_type_before_filter_dic:
                    summary_cell_type_before_filter_dic[col[1]] += 1
                else:
                    summary_cell_type_before_filter_dic[col[1]] = 1

                if float(col[2]) >= float(input_thr_value):
                    store_final_line.append(eachline)
                    store_filter_single_cell_id_dic[col[0]] = 1

                    if col[1] in summary_cell_type_after_filter_dic:
                        summary_cell_type_after_filter_dic[col[1]] += 1
                    else:
                        summary_cell_type_after_filter_dic[col[1]] = 1
            else:
                store_final_line.append(eachline)

    return (store_final_line,summary_cell_type_before_filter_dic,summary_cell_type_after_filter_dic,store_filter_single_cell_id_dic)

##store the remain id
def store_remain_id_in_exp_data (input_raw_exp_data_fl,store_filter_single_cell_id_dic):
    exp_fl_dt = pd.read_csv(input_raw_exp_data_fl, header=0, index_col=0)

    ##change the id to the list
    id_list = []
    for eachid in store_filter_single_cell_id_dic:
        id_list.append(eachid)
    exp_filter_id_dt = exp_fl_dt[id_list]

    return (exp_filter_id_dt)

