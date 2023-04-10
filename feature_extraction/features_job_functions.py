import argparse
from side_code.config import *
import os




def features_main_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_data_folder', action='store', type=str,
                        default=f"/Users/noa/Workspace/raxml_deep_learning_results/all_final_csvs_results")
    parser.add_argument('--results_folder', action='store', type=str,
                        default=RESULTS_FOLDER)
    parser.add_argument('--min_n_observations', action='store', type=int, default=1240)
    parser.add_argument('--iterations', action='store', type=int, default=40)
    parser.add_argument('--n_jobs', action='store', type=int, default=1)
    parser.add_argument('--jobs_prefix', action='store', type=str, default="fe_")
    parser.add_argument('--cpus_per_job', action='store', type=int, default=4)
    parser.add_argument('--queue', type=str, default="pupkolab")
    parser.add_argument('--perform_topology_tests', action='store_true', default=False)
    parser.add_argument('--msa_type', default='AA')
    parser.add_argument('--spr_iters',type=int, default=30)
    return parser


def feature_job_parser():
    parser = features_main_parser()
    parser.add_argument('--job_ind', action='store', type=int)
    parser.add_argument('--curr_job_raw_path', action='store', type=str)
    parser.add_argument('--curr_job_folder', action='store', type=str)
    parser.add_argument('--existing_msas_data', action='store', type=str)
    parser.add_argument('--features_output_path', action='store', type=str)
    parser.add_argument('--existing_msas_folder', action='store', type=str)
    return parser
