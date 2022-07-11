import sys

if sys.platform == "linux" or sys.platform == "linux2":
    PROJECT_ROOT_DIRECRTORY = "/groups/pupko/noaeker/RAxML_params_tuning"
else:
    PROJECT_ROOT_DIRECRTORY = "/Users/noa/Workspace/RAxML_params_tuning"
sys.path.append(PROJECT_ROOT_DIRECRTORY)

from side_code.file_handling import create_dir_if_not_exists, create_or_clean_dir, add_csvs_content
from side_code.config import *
from old_code.raw_data_aggregation_analysis.job_runner_side_funcs import get_sampling_job_related_files_paths, \
    data_sampling_parser
from side_code.code_submission import generate_argument_str, submit_linux_job, generate_argument_list, submit_local_job
from ML_pipeline.features_job_functions import features_main_parser
import pandas as pd
import pickle
import os
import numpy as np
import argparse
import random
from sklearn.model_selection import ParameterGrid


def generate_results_folder(curr_run_prefix):
    create_dir_if_not_exists(RESULTS_FOLDER)
    curr_run_prefix = os.path.join(RESULTS_FOLDER, curr_run_prefix)
    create_or_clean_dir(curr_run_prefix)
    return curr_run_prefix


def distribute_MSAS_over_jobs(raw_data, all_jobs_results_folder,args):
    jobs_csv_path_list = []
    msa_names = list(np.unique(raw_data["msa_path"]))
    msa_splits = np.array_split(list(msa_names), min(args.n_jobs, len(msa_names)))
    for job_ind, job_msas in enumerate(msa_splits):
        curr_job_folder = os.path.join(all_jobs_results_folder, "job_" + str(job_ind))
        create_or_clean_dir(curr_job_folder)
        current_raw_data_path = os.path.join(curr_job_folder, f"job_{job_ind}_raw_data{CSV_SUFFIX}")
        current_feature_output_path = os.path.join(curr_job_folder, f"job_{job_ind}_raw_data_with_features{CSV_SUFFIX}")
        jobs_csv_path_list.append(current_feature_output_path)
        current_raw_data = raw_data[raw_data["msa_path"].isin(job_msas)]
        current_raw_data.to_csv(current_raw_data_path, sep=CSV_SEP)

        run_command = f' python {FEATURE_EXTRACTION_CODE} --job_ind {job_ind} --curr_job_folder {curr_job_folder} --curr_job_raw_path {current_raw_data_path} --features_output_path {current_feature_output_path} {generate_argument_str(args)}' \
            f' --cpus_per_job {args.cpus_per_job}'

        if not LOCAL_RUN:
            job_name = args.jobs_prefix + str(job_ind)
            curr_job_log_path = os.path.join(curr_job_folder, str(job_ind) + "_tmp_log")
            submit_linux_job(job_name, curr_job_folder, curr_job_log_path, run_command, cpus=args.cpus_per_job,
                             job_ind=job_ind,
                             queue=args.queue)
        else:
            submit_local_job(FEATURE_EXTRACTION_CODE,
                             ["--job_ind", str(job_ind), "--curr_job_folder", curr_job_folder, "--curr_job_raw_path",
                              current_raw_data_path,
                              "--features_output_path", current_feature_output_path,
                              ]+ generate_argument_list(args))
    return jobs_csv_path_list


def main():
    parser = features_main_parser()
    args = parser.parse_args()
    feature_pipeline_dir = os.path.join(args.results_folder, "features_extraction_pipeline_files")
    create_dir_if_not_exists(feature_pipeline_dir)
    features_out_path = os.path.join(feature_pipeline_dir, f"all_features{CSV_SUFFIX}")
    all_jobs_running_folder = os.path.join(feature_pipeline_dir, "all_jobs")
    create_dir_if_not_exists(all_jobs_running_folder)
    existing_msas_data_path = os.path.join(feature_pipeline_dir, "all_msa_features")
    create_dir_if_not_exists(existing_msas_data_path)
    log_file_path = os.path.join(feature_pipeline_dir, "general_features.log")
    logging.basicConfig(filename=log_file_path, level=logging.DEBUG)
    csv_files_in_folder = [os.path.join(args.raw_data_folder, f) for f in
                           os.listdir(args.raw_data_folder) if f.endswith(CSV_SUFFIX)]
    dfs_in_folder = [pd.read_csv(f, sep=CSV_SEP) for f in csv_files_in_folder]
    logging.info(f"Combining CSV files: {csv_files_in_folder}")
    raw_data = pd.concat(dfs_in_folder, sort=False)
    counts = raw_data['msa_path'].value_counts()
    idx = counts[counts < args.min_n_observations].index
    raw_data = raw_data[~raw_data['msa_path'].isin(idx)]
    #if LOCAL_RUN:
    #    np.random.seed(SEED)
    #    msa_names = list(np.unique(raw_data["msa_path"]))
    #    msas_sample = np.random.choice(msa_names, size=3, replace=False)
    #    raw_data = raw_data[raw_data["msa_path"].isin(msas_sample)]
    jobs_csv_path_list = distribute_MSAS_over_jobs(raw_data, all_jobs_running_folder, args)
    prev_number_of_jobs_done = 0
    existing_csv_paths = []
    while len(existing_csv_paths) < len(jobs_csv_path_list):
        existing_csv_paths = [csv_path for csv_path in jobs_csv_path_list if os.path.exists(csv_path)]
        if len(existing_csv_paths) > prev_number_of_jobs_done:
            prev_number_of_jobs_done = len(existing_csv_paths)
            logging.info(f"total jobs done = {len(existing_csv_paths)}")
            add_csvs_content(jobs_csv_path_list, features_out_path)
    logging.info(f"done with all jobs! writing to csv in {features_out_path}")
    add_csvs_content(jobs_csv_path_list, features_out_path)


if __name__ == "__main__":
    main()
