#!/usr/bin/env python

##step 1: divide the cor value to multiple files and each cell type has one file
##step 2: compare corr value of each marker for each cell type and assign the cell type to marker
##step 3: divide the step 2 output into multiple cell type files and rank the features according to their correlation value
##step 4: generate top CORR markers and also generate CORR markers without overlapping with known markers

import sys
import re
import glob
import os
import pandas as pd
import os.path
from os import path


input_correlation_rate_fl = sys.argv[1]
##argument1: input1_correlation_rate_fl.txt

input_allknown_marker_gene_fl = sys.argv[2]
##argument2: input2_allknown_marker_gene_fl.txt

input_select_top_feat_num = sys.argv[3] ##select top 20
##argument3: 20

input_working_dir = sys.argv[4]
input_output_dir = sys.argv[5]

########
##step 1
def divide_to_cell_type_fl (input_correlation_rate_fl,input_working_dir):

    ##create a temp dir in the input_working_dir to stor ethe sorted corr file
    temp_sort_corr_dir = input_working_dir + '/step1_temp_sort_corr_dir'
    if not os.path.exists(temp_sort_corr_dir):
        os.makedirs(temp_sort_corr_dir)

    store_all_feat_dic = {}
    store_cell_type_dic = {}
    with open (input_correlation_rate_fl,'r') as ipt:
        for eachline in ipt:
            eachline = eachline.strip('\n')
            col = eachline.strip().split()
            store_cell_type_dic[col[1]] = 1
            store_all_feat_dic[col[0]] = 1

    for eachcell_type in store_cell_type_dic:

        store_final_line_list = []
        first_line = 'cell_name' + '\t' + 'feature' + '\t' + 'corr_value'
        store_final_line_list.append(first_line)
        with open(input_correlation_rate_fl, 'r') as ipt:
            for eachline in ipt:
                eachline = eachline.strip('\n')
                col = eachline.strip().split()

                if eachcell_type == col[1]:
                    final_line = eachcell_type + '\t' + col[0] + '\t' + col[2]
                    store_final_line_list.append(final_line)

        with open (temp_sort_corr_dir + '/opt_' + eachcell_type + '_corr.txt','w+') as opt:
            for eachline in store_final_line_list:
                opt.write(eachline + '\n')

        ##sort the corr value file
        corr_fl_dt = pd.read_table(temp_sort_corr_dir + '/opt_' + eachcell_type + '_corr.txt', delimiter=r"\s+") ##has header so we use header=None shut down
        ##sort the value
        cell_type_corr_fl_sort_dt = corr_fl_dt.sort_values(by='corr_value', ascending=False)
        ##save sorted dt to output
        cell_type_corr_fl_sort_dt.to_csv(temp_sort_corr_dir + '/opt_' + eachcell_type + '_sort.txt', index=False,sep='\t')  ##no index

    return (temp_sort_corr_dir,store_all_feat_dic,store_cell_type_dic)

########
##step 2
def assign_cell_type_to_feature (temp_sort_corr_dir,store_all_feat_dic,input_working_dir):

    store_cell_type_name_dic = {}
    store_final_assign_cell_type_line_list = []
    ft_id = 0
    for eachft in store_all_feat_dic:

        ft_id += 1
        print('analyze ft id is ' + str(ft_id))

        store_cell_type_corr_dic = {} ##each cell type has one corr value

        ##extract the corr for each cell type
        fl_list = glob.glob(temp_sort_corr_dir + '/*_sort.txt')
        for eachfl in fl_list:
            #print(eachfl)
            mt = re.match('.+/(.+)', eachfl)
            fl_nm = mt.group(1)
            mt = re.match('opt_(.+)_sort\.txt', fl_nm)
            cell_type_nm = mt.group(1)
            store_cell_type_name_dic[cell_type_nm] = 1

            corr_value = ''
            count = 0
            with open(eachfl, 'r') as ipt:
                for eachline in ipt:
                    eachline = eachline.strip('\n')
                    col = eachline.strip().split()

                    #print(eachline)

                    count += 1
                    if count != 1:  ##no column
                        gene_nm = col[1]

                        if eachft == gene_nm:
                            ##collect the corr

                            if len(col) == 3:
                                corr_value = col[2]
                                #print(corr_value)
                            else:
                                print('the line is wrong ' + eachline)

            if corr_value != '':
                store_cell_type_corr_dic[cell_type_nm] = float(corr_value)

            else:
                print('the corr value is empty ' + eachft)


        if len(list(store_cell_type_corr_dic.keys())) != 0:
            ##selec the key max as the cell type
            Keymax = max(store_cell_type_corr_dic, key=store_cell_type_corr_dic.get)

            final_line = Keymax + '\t' + eachft + '\t' + str(store_cell_type_corr_dic[Keymax])
            store_final_assign_cell_type_line_list.append(final_line)

    ##store the file a working_dir and divide the file into multiple files each files contains a cell type
    with open (input_working_dir + '/step2_temp_assign_cell_type_fl.txt','w+') as opt:
        for eachline in store_final_assign_cell_type_line_list:
            opt.write(eachline + '\n')


########
##step 3
def create_sort_corrvalue_allfeatures (input_working_dir,input_output_dir,temp_assign_cell_type_fl):

    ##create a dir in the working_dir to store the sorted unique fts in it.
    temp_assigned_corr_dir = input_working_dir + '/step3_temp_assigned_corr_dir'
    if not os.path.exists(temp_assigned_corr_dir):
        os.makedirs(temp_assigned_corr_dir)

    select_top_cell_type_sort_uni_fs_dir = input_output_dir + '/opt_1_assign_feature_with_celltype_dir'
    if not os.path.exists(select_top_cell_type_sort_uni_fs_dir):
        os.makedirs(select_top_cell_type_sort_uni_fs_dir)

    ##generate cell type dic
    store_cell_type_true_dic = {}
    with open(temp_assign_cell_type_fl, 'r') as ipt:
        for eachline in ipt:
            eachline = eachline.strip('\n')
            col = eachline.strip().split()
            cell_type = col[0]

            cell_type_nm = ''
            if cell_type == 'Meri..Xylem':
                cell_type_nm = 'Meri_Xylem'
            if cell_type == 'Phloem..CC.':
                cell_type_nm = 'Phloem_CC'
            if cell_type == 'Cortext':
                cell_type_nm = 'Cortex'
            if cell_type != 'Meri..Xylem' and cell_type != 'Phloem..CC.' and cell_type != 'Cortext':
                cell_type_nm = cell_type

            store_cell_type_true_dic[cell_type_nm] = 1

    ##divide the file to multiple
    for eachcell_type in store_cell_type_true_dic:

        store_final_line_list = []
        with open (temp_assign_cell_type_fl,'r') as ipt:
            for eachline in ipt:
                eachline = eachline.strip('\n')
                col = eachline.strip().split()
                cell_type = col[0]

                cell_type_nm = ''
                if cell_type == 'Meri..Xylem':
                    cell_type_nm = 'Meri_Xylem'
                if cell_type == 'Phloem..CC.':
                    cell_type_nm = 'Phloem_CC'
                if cell_type == 'Cortext':
                    cell_type_nm = 'Cortex'
                if cell_type != 'Meri..Xylem' and cell_type != 'Phloem..CC.' and cell_type != 'Cortext':
                    cell_type_nm = cell_type
                
                if eachcell_type == cell_type_nm:
                    store_final_line_list.append(eachline)

        with open (temp_assigned_corr_dir + '/temp_' + eachcell_type + '_corr.txt','w+') as opt:
            for eachline in store_final_line_list:
                opt.write(eachline + '\n')

        ##sort the temp corr file
        corr_fl_dt = pd.read_table(temp_assigned_corr_dir + '/temp_' + eachcell_type + '_corr.txt', header=None, delimiter=r"\s+")
        corr_fl_dt.columns = ['cell_name', 'feature', 'corr_value']
        cell_type_corr_fl_sort_dt = corr_fl_dt.sort_values(by='corr_value', ascending=False)
        cell_type_corr_fl_sort_dt.to_csv(select_top_cell_type_sort_uni_fs_dir + '/opt_' + eachcell_type + '_uni_fts_sort.txt', index=False,sep='\t')  ##no index


########
##step 4
def create_top_features (opt_1_assign_feature_with_celltype_dir,input_select_top_feat_num,input_marker_gene_fl):

    ##store the gene information
    store_marker_gene_dic = {}
    count = 0
    with open (input_marker_gene_fl,'r') as ipt:
        for eachline in ipt:
            eachline = eachline.strip('\n')
            count += 1
            if count != 1:
                col = eachline.strip().split()
                store_marker_gene_dic[col[0]] = 1


    store_top_feat_allcelltype_line_list = []
    store_top_feat_allcelltype_no_knownmarkers_line_list = []

    sort_fl_list = glob.glob(opt_1_assign_feature_with_celltype_dir + '/*')
    for eachsort_fl in sort_fl_list:
        mt = re.match('.+/(.+)', eachsort_fl)
        fl_nm = mt.group(1)
        mt = re.match('opt_(.+)_uni_fts_sort\.txt', fl_nm)
        cell_type = mt.group(1)

        cell_type_nm = ''
        if cell_type == 'Meri..Xylem':
            cell_type_nm = 'Meri_Xylem'
        if cell_type == 'Phloem..CC.':
            cell_type_nm = 'Phloem_CC'
        if cell_type == 'Cortext':
            cell_type_nm = 'Cortex'
        if cell_type != 'Meri..Xylem' and cell_type != 'Phloem..CC.' and cell_type != 'Cortext':
            cell_type_nm = cell_type


        corr_count = 0
        count = 0
        with open (eachsort_fl,'r') as ipt:
            for eachline in ipt:
                eachline = eachline.strip('\n')
                col = eachline.strip().split()

                count += 1
                if count != 1:
                    if count <= int(input_select_top_feat_num):
                        final_line = cell_type_nm + '\t' + col[1] + '\t' + col[2]
                        store_top_feat_allcelltype_line_list.append(final_line)

                    ##do not contain any known marker gene and select the top 20
                    if col[1] not in store_marker_gene_dic:
                        corr_count += 1
                        if corr_count <= int(input_select_top_feat_num):
                            store_top_feat_allcelltype_no_knownmarkers_line_list.append(eachline)

    with open (input_output_dir + '/opt_2_top_' + input_select_top_feat_num + '_corr.txt','w+') as opt:
        opt.write('Celltype' + '\t' + 'Marker' + '\t' + 'Corr_value' + '\n')
        for eachline in store_top_feat_allcelltype_line_list:
            opt.write(eachline + '\n')

    with open (input_output_dir + '/opt_3_top_' + input_select_top_feat_num + '_corr_without_known_markers.txt','w+') as opt:
        opt.write('Celltype' + '\t' + 'Marker' + '\t' + 'Corr_value' + '\n')
        for eachline in store_top_feat_allcelltype_no_knownmarkers_line_list:
            opt.write(eachline + '\n')

print ('step1: Divide files')
temp_sort_corr_dir,store_all_feat_dic,store_cell_type_dic = divide_to_cell_type_fl (input_correlation_rate_fl,input_working_dir)

print ('step2: assign cell type')
if path.exists(input_working_dir + '/step2_temp_assign_cell_type_fl.txt'):
    print('step2_temp_assign_cell_type_fl.txt exists. Go to step3')
else:
    assign_cell_type_to_feature (temp_sort_corr_dir,store_all_feat_dic,input_working_dir)

print ('step3: create all features')
create_sort_corrvalue_allfeatures (input_working_dir,input_output_dir,input_working_dir + '/step2_temp_assign_cell_type_fl.txt')

print ('step4: create top features')
create_top_features (input_output_dir + '/opt_1_assign_feature_with_celltype_dir',input_select_top_feat_num,input_allknown_marker_gene_fl)



