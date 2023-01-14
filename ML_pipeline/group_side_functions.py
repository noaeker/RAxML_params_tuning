import argparse
from side_code.config import *
import os
import numpy as np




def group_main_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', action='store', type=str,
                        default="/Users/noa/Workspace/raxml_deep_learning_results/test_single_tree_data_err.tsv")
    parser.add_argument('--curr_working_dir',
                        default="/Users/noa/Workspace/raxml_deep_learning_results/new_grouping_test")
    parser.add_argument('--n_pars_trees', action='store', type=int, default=0)
    parser.add_argument('--n_rand_trees', action='store', type=int, default=5)
    parser.add_argument('--n_iterations', action='store', type=int, default=2)
    parser.add_argument('--n_pars_trees_sample', type=int, default=50)
    parser.add_argument('--name', type=str, default="groups_run_5_5")
    parser.add_argument('--filter_on_default_data', action='store_true', default = True)
    #parser.add_argument('--large_grid', action='store_true', default=False)
    parser.add_argument('--n_jobs', type=int, default=3)
    parser.add_argument('--n_cv_folds', type=int, default=5)
    parser.add_argument('--cpus_per_job', type = int, default=1)
    parser.add_argument('--cpus_per_main_job', type=int, default=4)
    parser.add_argument('--level', type=str, default='info')
    parser.add_argument('--queue', type=str, default='power-pupko')
    parser.add_argument('--jobs_prefix', type = str, default = 'groups_job'
                        )
    parser.add_argument('--include_output_tree_features', action = 'store_true', default= True)
    parser.add_argument('--sample_fracs', default = [1] )#np.linspace(0.1,1,5)
    return parser


def group_job_parser():
    parser = group_main_parser()
    parser.add_argument('--job_ind', action='store', type=int)
    parser.add_argument('--curr_job_raw_path', action='store', type=str)
    parser.add_argument('--curr_job_folder', action='store', type=str)
    parser.add_argument('--curr_job_group_output_path', action='store', type=str)
    return parser
