from side_code.file_handling import create_dir_if_not_exists,create_or_clean_dir
from side_code.config import *
from raw_data_sampling_analysis.job_runner_side_funcs import get_sampling_job_related_files_paths, data_sampling_parser
from side_code.code_submission import generate_argument_str,submit_linux_job,generate_argument_list,submit_local_job
import pandas as pd
import pickle
import os
import numpy as np
import random
from sklearn.model_selection import ParameterGrid



def generate_results_folder(curr_run_prefix):
    create_dir_if_not_exists(RESULTS_FOLDER)
    curr_run_prefix = os.path.join(RESULTS_FOLDER, curr_run_prefix)
    create_or_clean_dir(curr_run_prefix)
    return curr_run_prefix


def distribute_grid_over_jobs(parameter_grid, all_jobs_results_folder, args):
    jobs_csv_path_list = []
    status_file_path_list = []
    grid_splits = np.array_split(list(parameter_grid), args.n_jobs)
    for job_ind, job_grid in enumerate(grid_splits):
        curr_job_folder = os.path.join(all_jobs_results_folder, "job_" + str(job_ind))
        create_or_clean_dir(curr_job_folder)

        job_related_files_paths = get_sampling_job_related_files_paths(curr_job_folder, job_ind)
        jobs_csv_path_list.append(job_related_files_paths["job_csv_path"])
        status_file_path_list.append(job_related_files_paths["job_status_file"])
        pickle.dump(job_grid, open(job_related_files_paths["job_grid_points_file"], "wb"))
        run_command = f' python {SAMPLING_MAIN_CODE_PATH} ' \
            ' --job_ind {job_ind} --curr_job_folder {curr_job_folder} {previous_args}' \
            .format(
            job_ind=job_ind, previous_args=generate_argument_str(args), curr_job_folder=curr_job_folder
        )
        job_name = args.jobs_prefix + str(job_ind)
        if not LOCAL_RUN:
            submit_linux_job(job_name, curr_job_folder, run_command, 1, job_ind, queue=args.queue)
        else:
            submit_local_job(SAMPLING_MAIN_CODE_PATH, ["--job_ind", str(job_ind), "--curr_job_folder", curr_job_folder
                                              ] + generate_argument_list(args))
    csv_path_to_status_path_dict = {csv_path: status_path for csv_path, status_path in
                                    zip(jobs_csv_path_list, status_file_path_list)}
    return csv_path_to_status_path_dict


def main():
    parser = data_sampling_parser()
    args = parser.parse_args()
    all_jobs_results_folder = generate_results_folder(args.run_prefix)
    all_jobs_general_log_file = os.path.join(all_jobs_results_folder, "log_file.log")
    logging.basicConfig(filename=all_jobs_general_log_file, level=LOGGING_LEVEL)
    logging.info("Args = {args}".format(args=args))
    random.seed(SEED)
    args = parser.parse_args()
    data = pd.read_csv(args.raw_data_path, sep=CSV_SEP)
    grid_data = data[data["run_name"] != "default"]
    random_spr_cutoff_options = np.unique(grid_data["spr_cutoff"])
    random_spr_radius_options = np.unique(grid_data["spr_radius"])
    parsimony_spr_cutoff_options = random_spr_cutoff_options.copy()
    parsimony_spr_radius_options = random_spr_radius_options.copy()
    parameter_grid = ParameterGrid([
        {"random_spr_cutoff": random_spr_cutoff_options, "random_spr_radius": random_spr_radius_options,
         "parsimony_spr_cutoff": parsimony_spr_cutoff_options, "parsimony_spr_radius": parsimony_spr_radius_options,
         "n_random": range(args.n_random + 1), "n_parsimony": range(1,args.n_parsimony + 1)},

        {"random_spr_cutoff": random_spr_cutoff_options, "random_spr_radius": random_spr_radius_options,
         "parsimony_spr_cutoff": parsimony_spr_cutoff_options, "parsimony_spr_radius": parsimony_spr_radius_options,
         "n_random": range(1,args.n_random + 1), "n_parsimony": range(args.n_parsimony + 1)}]
    )
    logging.info(f"Grid size is: {len(list(parameter_grid))}"
    )
    distribute_grid_over_jobs(parameter_grid, all_jobs_results_folder, args)


if __name__ == "__main__":
    main()
