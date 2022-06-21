from side_code.config import *
import argparse
import os

def data_sampling_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_data_path', action='store', type=str, default=f"{RESULTS_FOLDER}/full_raxml_data.tsv")
    parser.add_argument('--n_parsimony', action='store', type=int,
                        default=10)
    parser.add_argument('--n_random', action='store', type=int,
                        default=10)
    parser.add_argument('--n_sample_points', action='store', type=int,
                        default=15)
    parser.add_argument('--n_jobs', action='store', type=int,
                        default=2)
    parser.add_argument('--seed', action='store', type=int,
                        default=SEED)
    parser.add_argument('--run_prefix', action='store', type=str,
                        default="random_parismony_sampling")
    parser.add_argument('--jobs_prefix', action='store', type=str, default="random_parismony_sampling_jobs")
    parser.add_argument('--queue', type=str, default="pupkolab")
    return parser

def random_and_parsimony_job_parser():
    parser = data_sampling_parser()
    parser.add_argument('--job_ind', action='store', type=int)
    parser.add_argument('--curr_job_folder', action='store', type=str)
    return parser

def get_sampling_job_related_files_paths(curr_job_folder, job_ind):
    job_status_file = os.path.join(curr_job_folder, str(job_ind) + "_status")
    job_csv_path = os.path.join(curr_job_folder, str(job_ind) + CSV_SUFFIX)
    job_grid_points_file = os.path.join(curr_job_folder, "grid_points_" + str(job_ind))
    general_log_path = os.path.join(curr_job_folder, "job_" + str(job_ind) + "_general_log.log")
    return {"job_status_file": job_status_file, "job_csv_path": job_csv_path,
            "job_grid_points_file": job_grid_points_file,
            "general_log_path": general_log_path}
