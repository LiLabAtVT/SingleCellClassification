#!/usr/bin/env python

##this script will use the Rscript to
##conduct analysis on the Step 1 to 3.
##1. prepare data
##2. train models
##3. identify markers

import os
import argparse
import sys
import subprocess

def get_parsed_args():

    parser = argparse.ArgumentParser(description="SPmarker prepare data")

    ##require files
    parser.add_argument("-d", dest='working_dir', default="./", help="Working directory to store intermediate files of "
                                                                     "each step. Default: ./ ")

    parser.add_argument("-o", dest='output_dir', default="./", help="Output directory to store the output files."
                                                                    "Default: ./ ")

    parser.add_argument("-SPmarker_dir" ,dest="SPmarker_directory",help="Provide the path to the SPmarker_directory")

    parser.add_argument("-merged_obj", dest="merged_object", help="Provide merged object generated from Seurat.")

    parser.add_argument("-R_p", dest="R_path", help="Provide Rscript path."
                                                    "Default: /usr/bin/Rscript.")

    ##Optional parameters
    parser.add_argument("-kmar_fl", dest="known_marker_fl", help="Provide the known marker gene list file. Once users provide this file, "
                                                                 "they will obtain a file that contains novel marker genes.")

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
    if args.SPmarker_directory is None:
        print ('Cannot find SPmarker_directory, please provide that')
        return

    if args.R_path is None:
        R_path = '/usr/bin/Rscript'
    else:
        R_path = args.R_path

    if args.merged_object is None:
        print('Cannot find merged object, please provide it')
        return
    else:
        try:
            file = open(args.merged_object, 'r')  ##check if the file is not the right file
        except IOError:
            print('There was an error opening the merged object file!')
            return

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
    SPmarker_directory = args.SPmarker_directory
    merged_object = args.merged_object

    ########
    ##Step 1 prepare data
    print('Step 1 prepare data')

    Step1_prepare_data_path = SPmarker_directory + '/Step1_prepare_data.py'

    Step1_prepare_data_dir = working_dir + '/Step1_prepare_data_dir'
    if not os.path.exists(Step1_prepare_data_dir):
        os.makedirs(Step1_prepare_data_dir)

    Step1_prepare_data_w_dir = Step1_prepare_data_dir + '/Step1_prepare_data_w_dir'
    if not os.path.exists(Step1_prepare_data_w_dir):
        os.makedirs(Step1_prepare_data_w_dir)

    Step1_prepare_data_o_dir = Step1_prepare_data_dir + '/Step1_prepare_data_o_dir'
    if not os.path.exists(Step1_prepare_data_o_dir):
        os.makedirs(Step1_prepare_data_o_dir)

    ##run the process of Step1
    cmd = 'python ' + Step1_prepare_data_path + \
          ' -d ' + Step1_prepare_data_w_dir + \
          ' -o ' + Step1_prepare_data_o_dir + \
          ' -sup_dir ' + SPmarker_directory + '/sup_dir' + \
          ' -merged_obj ' + merged_object + \
          ' -R_p ' + R_path + \
          ' -ICI yes' + \
          ' -itg yes' + \
          ' -fl_feat yes'
    print(cmd)
    subprocess.call(cmd,shell=True)

    ########
    ##Step 2 train models
    print('Step 2 train models')

    Step2_train_models_path = SPmarker_directory + '/Step2_train_models.py'

    Step2_train_models_dir = working_dir + '/Step2_train_models_dir'
    if not os.path.exists(Step2_train_models_dir):
        os.makedirs(Step2_train_models_dir)

    Step2_train_models_w_dir = Step2_train_models_dir + '/Step2_train_models_w_dir'
    if not os.path.exists(Step2_train_models_w_dir):
        os.makedirs(Step2_train_models_w_dir)

    Step2_train_models_o_dir = Step2_train_models_dir + '/Step2_train_models_o_dir'
    if not os.path.exists(Step2_train_models_o_dir):
        os.makedirs(Step2_train_models_o_dir)

    ##run the process of Step2
    cmd = 'python ' + Step2_train_models_path + \
          ' -d ' + Step2_train_models_w_dir + \
          ' -o ' + Step2_train_models_o_dir + \
          ' -ICI_fl ' + Step1_prepare_data_o_dir + '/ICIn.csv' + \
          ' -exp_fl ' + Step1_prepare_data_o_dir + '/merged_data_after_integration_mtx.csv' + \
          ' -feat_fl ' + Step1_prepare_data_o_dir + '/var_*_feats.csv'
    print(cmd)
    subprocess.call(cmd,shell=True)

    ########
    ##Step 3 identify marker
    print('Step 3 identify marker')

    Step3_identify_marker_path = SPmarker_directory + '/Step3_identify_marker.py'

    Step3_identify_marker_dir = working_dir + '/Step3_identify_marker_dir'
    if not os.path.exists(Step3_identify_marker_dir):
        os.makedirs(Step3_identify_marker_dir)

    Step3_identify_marker_w_dir = working_dir + '/Step3_identify_marker_w_dir'
    if not os.path.exists(Step3_identify_marker_w_dir):
        os.makedirs(Step3_identify_marker_w_dir)

    if args.known_marker_fl is not None:
        known_marker_fl = args.known_marker_fl
        cmd = 'python ' + Step3_identify_marker_path + \
              ' -d ' + Step3_identify_marker_w_dir + \
              ' -o ' + output_dir + \
              ' -m ' + Step2_train_models_o_dir + '/rf_model.pkl' + \
              ' -exp_fl ' + Step2_train_models_o_dir + '/opt_exp_indep_test.csv' + \
              ' -meta_fl ' + Step2_train_models_o_dir + '/opt_meta_indep_test.csv' + \
              ' -kmar_fl ' + known_marker_fl
        print(cmd)
        subprocess.call(cmd,shell=True)

    else:
        cmd = 'python ' + Step3_identify_marker_path + \
              ' -d ' + Step3_identify_marker_w_dir + \
              ' -o ' + output_dir + \
              ' -m ' + Step2_train_models_o_dir + '/rf_model.pkl' + \
              ' -exp_fl ' + Step2_train_models_o_dir + '/opt_exp_indep_test.csv' + \
              ' -meta_fl ' + Step2_train_models_o_dir + '/opt_meta_indep_test.csv'
        print(cmd)
        subprocess.call(cmd,shell=True)



if __name__ == "__main__":
    main()



