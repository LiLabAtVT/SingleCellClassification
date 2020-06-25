#!/usr/bin/env python

##this script will use the Rscript to
##1) integrate data or not
##2) assign cell type using ICI or not
##3) generate filtered feature

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

    parser.add_argument("-sup_dir" ,dest="sup_directory",help="Provide supplementary directory provided in this pipeline"
                                                              "This supplementary dir contains the Rscript used in this step.")

    parser.add_argument("-merged_obj", dest="merged_object", help="Provide merged object generated from Seurat.")

    parser.add_argument("-R_p", dest="R_path", help="Provide Rscript path."
                                                    "Default: /usr/bin/Rscript.")

    ##optional parameters
    parser.add_argument("-itg" ,dest="itg_yes", help="Provide the yes, if users want to conduct integration using the Seurat.")

    #parser.add_argument("-f_num", dest="feature_number", help="If users initiate the -itg, please provide the feature number in the merged object.")
    parser.add_argument("-dim_num" ,dest='dim_number', help="If users initiate the -itg, please provide the npcs number."
                                                              "Defaul: 30")

    parser.add_argument("-ICI", dest="ICI_yes", help="Provide the yes, if users want to use ICI to assign the cells with cell type.")

    parser.add_argument("-fl_feat", dest="fl_feat_yes", help="Provide the yes, if users want to conduct feature selection in the Seurat")

    parser.add_argument("-fl_pro", dest="fl_feat_proportion", help="Provide the proportion of features users want to keep."
                                                                   "For example, if the fl_pro is set to 0.2 when we have 100 features."
                                                                   "Finally, we will obtain 20 features with the most variable expression."
                                                                   "Default: 0.2")
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
    if args.sup_directory is None:
        print ('Cannot find supplementary dir, please provide that')
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

    if args.dim_number is not None:
        dim_number = args.dim_number
    else:
        dim_number = '30'

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
    ##Step 0 data integration
    sup_directory = args.sup_directory
    merged_object = args.merged_object

    if args.itg_yes is not None:
        if args.itg_yes != 'yes':
            print('Please use yes to call this argument')
            return
        else:
            print('Step 0 data integration')
            ##Begin analysis
            ##generate working and output dir
            S0_data_integration_dir = working_dir + '/S0_data_integration_dir'
            if not os.path.exists(S0_data_integration_dir):
                os.makedirs(S0_data_integration_dir)

            S0_data_integration_w_dir = S0_data_integration_dir + '/S0_data_integration_w_dir'
            if not os.path.exists(S0_data_integration_w_dir):
                os.makedirs(S0_data_integration_w_dir)

            integration_r_script = sup_directory + '/Step1_0_Integration.R'

            cmd = R_path + ' ' + integration_r_script + ' ' + merged_object + ' ' + \
                  dim_number + ' ' + output_dir + ' ' + S0_data_integration_w_dir
            print(cmd)
            subprocess.call(cmd, shell=True)

    ########
    ##Step 1 assign cell type to cells using the ICI
    if args.ICI_yes is not None:

        if args.ICI_yes != 'yes':
            print('Please use yes to call this argument')
            return
        else:

            print('Step 1 assign cell type to cells using the ICI index method')

            S1_assign_cell_type_dir = working_dir + '/S1_assign_cell_type_dir'
            if not os.path.exists(S1_assign_cell_type_dir):
                os.makedirs(S1_assign_cell_type_dir)

            S1_assign_cell_type_w_dir = S1_assign_cell_type_dir + '/S1_assign_cell_type_w_dir'
            if not os.path.exists(S1_assign_cell_type_w_dir):
                os.makedirs(S1_assign_cell_type_w_dir)

            assign_cell_type_r_script = sup_directory + '/Step1_1_Assign_cell_type.R'
            spec_file = sup_directory + '/ath_root_marker_spec_score.csv'

            cmd = R_path + ' ' + assign_cell_type_r_script + ' ' + merged_object + ' ' + \
                  spec_file + ' ' + output_dir + ' ' + S1_assign_cell_type_w_dir
            print(cmd)
            subprocess.call(cmd,shell=True)

    ########
    ##Step 2 filter features
    if args.fl_feat_yes is not None:
        if args.fl_feat_yes != 'yes':
            print('Please use yes to call this argument')
        else:
            if args.fl_feat_proportion is not None:
                fl_feat_proportion = args.fl_feat_proportion
            else:
                fl_feat_proportion = '0.2'

            print('Step 2 filter features')

            filter_feature_r_script = sup_directory + '/Step1_2_Filter_features.R'

            ##two cases:
            ##case one, users do not integrate the object
            ##case two, users integrate the object
            ##merged_data will be used to fitler features
            if args.itg_yes is None:
                ipt_data_obj_rds = merged_object
            else:
                ipt_data_obj_rds = working_dir + '/S0_data_integration_dir/S0_data_integration_w_dir/merged_data_after_integration_obj.rds'

            ##run the process
            cmd = R_path + ' ' + filter_feature_r_script + ' ' + ipt_data_obj_rds + ' ' + fl_feat_proportion + ' ' + output_dir
            print(cmd)
            subprocess.call(cmd,shell=True)


if __name__ == "__main__":
    main()



