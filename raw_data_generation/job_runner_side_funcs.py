import argparse
from side_code.config import *
import os
import time


def main_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_prefix', action='store', type=str, default=CURR_RUN_PREFIX)
    parser.add_argument('--general_msa_dir',type=str, default = GENERAL_MSA_DIR)
    parser.add_argument('--jobs_prefix', action='store', type=str, default=CURR_JOBS_PREFIX)
    parser.add_argument('--logging_level', action='store', type = str,default  = LOGGING_LEVEL)
    parser.add_argument('--n_MSAs', action='store', type=int, default=N_MSAS)
    parser.add_argument('--run_raxml_commands_locally',action='store_true', default= True)
    parser.add_argument('--first_msa_ind', action='store', type=int, default=0)
    parser.add_argument('--min_n_seq', action='store', type=int, default=MIN_N_SEQ)
    parser.add_argument('--max_n_seq', action='store', type=int, default=MAX_N_SEQ)
    parser.add_argument('--min_n_loci', type=int, default=MIN_N_LOCI)
    parser.add_argument('--n_raxml_parsimony_trees', action='store', type=int, default=N_PARSIMONY_RAXML_SEARCH)
    parser.add_argument('--n_raxml_random_trees', action='store', type=int, default=N_RANDOM_RAXML_SEARCH)
    parser.add_argument('--use_raxml_search', action='store_true', default= True)  # change
    parser.add_argument('--queue', type=str, default="public")
    parser.add_argument('--n_cpus_per_job', action='store', type=int, default=N_CPUS_PER_JOB)
    parser.add_argument('--n_cpus_raxml', action='store', type=int, default=N_CPUS_RAXML)
    parser.add_argument('--spr_radius_grid', action='store', type=str, default=SPR_RADIUS_GRID)
    parser.add_argument('--spr_cutoff_grid', action='store', type=str, default=SPR_CUTOFF_GRID)
    parser.add_argument('--trim_msa',action='store_true')
    parser.add_argument('--remove_output_files',action='store_true', default= True)
    parser.add_argument('--use_existing_global_data',action='store_true')
    parser.add_argument('--existing_global_data_to_use', type=str, default = "pandit_tuning")
    parser.add_argument('--max_n_parallel_jobs', action='store', type=int, default=N_JOBS)
    parser.add_argument('--print_commands_to_log', action='store_true')
    parser.add_argument('--n_iters_test', action='store', type=int, default=TEST_MSA_ITERATIONS)
    parser.add_argument('--n_MSAs_per_bunch',type=int, default= N_MSAS_PER_BUNCH)
    parser.add_argument('--MSAs_pool_size', type=int, default=MSAs_POOL_SIZE)
    parser.add_argument('--use_files_from_csv', action='store_true')
    parser.add_argument('--timeout',type=int, default = 1000)
    parser.add_argument('--max_n_tasks_per_job',type=int, default=2)
    parser.add_argument('--waiting_time_between_iterations',type=int, default = WAITING_TIME_UPDATE)
    parser.add_argument('--waiting_time_between_job_submissions', type=int, default=4)
    parser.add_argument('--msa_type', default = 'AA',type=str)
    parser.add_argument('--old_msas_folder', type=str)
    parser.add_argument('--test_msa_type', default = 'AA')
    parser.add_argument('--time_between_tests',type=int, default=300)
    parser.add_argument('--fake_run', type = bool, default= True)
    return parser


def job_parser():
    parser = main_parser()
    parser.add_argument('--job_ind', action='store', type=int)
    parser.add_argument('--curr_job_folder', action='store', type=str)
    parser.add_argument('--test_msa', action='store', type=str)
    parser.add_argument('--current_tasks_path',action='store', type=str)
    return parser


def get_job_related_files_paths(curr_job_folder, job_ind):
    job_status_file = os.path.join(curr_job_folder, str(job_ind) + "_status")
    job_local_tasks_path = os.path.join(curr_job_folder, "job_local_tasks_path" + str(job_ind))
    job_local_raxml_done_run = os.path.join(curr_job_folder, "local_raxml_done" + str(job_ind))
    job_local_stop_running_path = os.path.join(curr_job_folder, "local_raxml_stop" + str(job_ind))
    general_log_path = os.path.join(curr_job_folder, "job_" + str(job_ind) + "_general_log.log")
    job_log_folder = os.path.join(curr_job_folder,f'{job_ind}_tmp_log')
    return {"job_status_file": job_status_file, "job_local_tasks_path": job_local_tasks_path,
            "job_local_done_dump": job_local_raxml_done_run,
            "job_local_stop_running_path": job_local_stop_running_path,
            "job_log_path": general_log_path,
            "job_log_folder": job_log_folder,
            "job_entire_folder": curr_job_folder,
            "job_started_file":os.path.join(curr_job_folder,"started")
            }



