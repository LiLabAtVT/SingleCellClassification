#!/usr/bin/env python

import re

def extract_marker_gene (input_exp_dt,input_variable_gene_list_fl):

    ##store the gene information that will be filtered out
    store_variable_gene_dic = {}
    count = 0
    with open (input_variable_gene_list_fl,'r') as ipt:
        for eachline in ipt:
            eachline = eachline.strip('\n')
            count += 1
            if count != 1:
                col = eachline.strip().split(',')
                mt = re.match('"(.+)"',col[1])
                feat = mt.group(1)
                store_variable_gene_dic[feat] = 1

    exp_line_list = []
    count = 0
    with open (input_exp_dt,'r') as ipt:
        for eachline in ipt:
            eachline = eachline.strip('\n')
            count += 1
            if count != 1:
                col = eachline.strip().split(',')

                mt = re.match('"(.+)"', col[0])
                feat = mt.group(1)

                ##updation change in the gene dic
                if feat in store_variable_gene_dic:
                    exp_line_list.append(eachline)
            else:
                exp_line_list.append(eachline)

    return (exp_line_list)

