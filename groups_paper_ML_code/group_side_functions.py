import argparse
from side_code.config import *
import os
import numpy as np




def group_main_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_data_folder', action='store', type=str,
                        default="/Users/noa/Workspace/raxml_deep_learning_results/ds_data")
    parser.add_argument('--curr_working_dir',
                        default="/Users/noa/Workspace/raxml_deep_learning_results/new_grouping_test")
    parser.add_argument('--n_pars_trees', action='store', type=int, default=-1)
    parser.add_argument('--n_rand_trees', action='store', type=int, default=-1)
    parser.add_argument('--n_sum_range',action='store', type=str, default = '1_40')
    parser.add_argument('--n_iterations', action='store', type=int, default=1)
    #parser.add_argument('--n_pars_trees_sample', type=int, default=50)
    parser.add_argument('--name', type=str, default="groups_run_local_test")
    parser.add_argument('--filter_on_default_data', action='store_true', default = True)
    #parser.add_argument('--large_grid', action='store_true', default=False)
    parser.add_argument('--n_jobs', type=int, default=3)
    parser.add_argument('--n_cv_folds', type=int, default=2)
    parser.add_argument('--cpus_per_job', type = int, default=1)
    parser.add_argument('--cpus_per_main_job', type=int, default=4)
    parser.add_argument('--level', type=str, default='info')
    parser.add_argument('--queue', type=str, default='power-pupko')
    parser.add_argument('--jobs_prefix', type = str, default = 'groups_job'
                        )
    parser.add_argument('--include_output_tree_features', action = 'store_true', default= True)
    parser.add_argument('--sample_fracs', default =np.linspace(0.1,1,7) )#np.linspace(0.1,1,5)#np.linspace(0.1,1,7)
    parser.add_argument('--model', default = "lightgbm") #sgd,rf ,lightgbm
    parser.add_argument('--add_sample_fracs',action = 'store_true')
    parser.add_argument('--additional_validation', default=None)#"/Users/noa/Workspace/raxml_deep_learning_results/new_grouping_test/groups_run_local_test/group_results_20_subsample.tsv"
    parser.add_argument('--prev_results_path', default = None)
    parser.add_argument('--program', type=str, default='RAxML')
    parser.add_argument('--simulated', action='store_true', default=False)
    parser.add_argument('--msa_type',type=str, default= 'AA')
    parser.add_argument('--ll_epsilon', type = str, default = '0')
    parser.add_argument('--ML_epsilon', type = float, default = 0)
    parser.add_argument('--external_best_tree_file',type=str, default='Nope')
    parser.add_argument('--only_generate_data', action= 'store_true',default=False)
    parser.add_argument('--perform_topology_tests', action = 'store_true', default= True )
    parser.add_argument('--only_validation', action = 'store_true', default= True )
    parser.add_argument('--n_validation', type = int, default = 200)
    parser.add_argument('--external_validation_data', type = str, default= 'Nope')
    return parser


def group_job_parser():
    parser = group_main_parser()
    parser.add_argument('--job_ind', action='store', type=int)
    parser.add_argument('--curr_job_raw_path', action='store', type=str)
    parser.add_argument('--curr_job_folder', action='store', type=str)
    parser.add_argument('--curr_job_group_output_path', action='store', type=str)
    parser.add_argument('--curr_job_group_output_raw_path', action='store', type=str)

    return parser
