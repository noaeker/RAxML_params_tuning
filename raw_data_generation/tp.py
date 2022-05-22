import sys

if sys.platform == "linux" or sys.platform == "linux2":
    PROJECT_ROOT_DIRECRTORY = "/groups/pupko/noaeker/RAxML_params_tuning"
else:
    PROJECT_ROOT_DIRECRTORY = "/Users/noa/Workspace/RAxML_params_tuning"
sys.path.append(PROJECT_ROOT_DIRECRTORY)

from msa_runs import generate_all_raxml_runs_per_msa
from side_code.config import *
from side_code.code_submission import submit_linux_job, submit_local_job, generate_argument_list,generate_argument_str
from side_code.file_handling import create_dir_if_not_exists, create_or_clean_dir, extract_alignment_files_from_dirs
from side_code.code_submission import is_job_done
from side_code.MSA_manipulation import remove_MSAs_with_not_enough_seq_and_locis, trim_MSA, get_alignment_data
from job_runner_side_funcs import main_parser, get_job_related_files_paths
from side_code.file_handling import unify_dicts
import pickle
import os
import time
from shutil import rmtree
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import uuid
from math import ceil
import subprocess
from subprocess import PIPE, STDOUT, call
import sys


def generate_results_folder(curr_run_prefix):
    create_dir_if_not_exists(RESULTS_FOLDER)
    curr_run_prefix = os.path.join(RESULTS_FOLDER, curr_run_prefix)
    create_or_clean_dir(curr_run_prefix)
    return curr_run_prefix


def submit_single_job(all_jobs_results_folder, job_ind, curr_job_tasks_list, test_msa_path, current_tasks_path, args):
    curr_job_folder = os.path.join(all_jobs_results_folder, "job_" + str(job_ind))
    create_or_clean_dir(curr_job_folder)
    curr_job_related_files_paths = get_job_related_files_paths(curr_job_folder, job_ind)
    curr_job_tasks_path = curr_job_related_files_paths["job_local_tasks_path"]
    pickle.dump(curr_job_tasks_list, open(curr_job_tasks_path, "wb"))
    curr_job_log_path = os.path.join(curr_job_folder, str(job_ind) + "_tmp_log")
    run_command = f' python {MAIN_CODE_PATH} --job_ind {job_ind} --curr_job_folder {curr_job_folder} --test_msa {test_msa_path} --current_tasks_path {current_tasks_path} {generate_argument_str(args)} '
    job_name = f"{job_ind}_{args.jobs_prefix}"
    if not LOCAL_RUN:
        submit_linux_job(job_name, curr_job_folder, curr_job_log_path, run_command, args.n_cpus_per_job, job_ind,
                         queue=args.queue)
    else:
        submit_local_job(MAIN_CODE_PATH,
                         ["--job_ind", str(job_ind), "--curr_job_folder", curr_job_folder, "--test_msa", test_msa_path,"--current_tasks_path", current_tasks_path
                          ] + generate_argument_list(args))
    return curr_job_related_files_paths


def generate_file_path_list_and_test_msa(args, trimmed_test_msa_path):
    file_path_list = extract_alignment_files_from_dirs(args.general_msa_dir)
    if args.MSAs_pool_size > 0:
        file_path_list = file_path_list[:args.MSAs_pool_size]
    logging.info("There are overall {nMSAs} available MSAs ".format(nMSAs=len(file_path_list)))
    file_path_list_full = remove_MSAs_with_not_enough_seq_and_locis(file_path_list, args.min_n_seq, args.min_n_loci)
    test_msa_path = file_path_list_full[0]
    trim_MSA(test_msa_path, trimmed_test_msa_path, number_of_sequences=10, max_n_loci=500, loci_shift=0)
    logging.debug("Alignment files are " + str(file_path_list))
    random.seed(SEED)
    file_path_list = random.sample(file_path_list_full, min(args.n_MSAs, len(file_path_list_full)))
    logging.info(
        "There are {} MSAs with at least {} sequences and {} positions".format(len(file_path_list), args.min_n_seq,
                                                                               args.min_n_loci))
    logging.info(
        f"Sampling {len(file_path_list)} random MSAs")
    return file_path_list


def  update_results_tasks_and_jobs(job_tracking_dict, global_results_path, current_tasks_path, global_results_csv_path):
    for job_ind in list(job_tracking_dict.keys()):
        if  os.path.exists(job_tracking_dict[job_ind]["job_local_done_dump"]):
            job_raxml_runs_done_obj = pickle.load(open(job_tracking_dict[job_ind]["job_local_done_dump"], "rb"))
            logging.debug(f"Job done size is {len(job_raxml_runs_done_obj)}")
            # update global results
            global_results_dict = pickle.load(open(global_results_path, "rb"))
            global_results_dict.update(job_raxml_runs_done_obj)  # update new results
            pickle.dump(global_results_dict, open(global_results_path, "wb"))
            global_results_to_csv(global_results_dict,global_results_csv_path)
            logging.debug(f"Global results dict size is now {len(global_results_dict)}")
            # update tasks dictionary
            tasks_dict = pickle.load(open(current_tasks_path, "rb"))
            tasks_dict = {task_ind: tasks_dict[task_ind] for task_ind in tasks_dict if
                          task_ind not in job_raxml_runs_done_obj}  # insert leftover tasks
            logging.debug(f"Tasks dict size is now {len(tasks_dict)}")
            pickle.dump(tasks_dict, open(current_tasks_path, "wb"))
            # remove job tracking dict is job is done
            if is_job_done(job_tracking_dict[job_ind]["job_log_folder"]): #if job is done, remove it from dictionary
                logging.info(f"Job {job_ind} is done, global results size is now {len(global_results_dict)}, time = {time.strftime('%m/%d/%Y, %H:%M:%S', time.localtime())}")
                if os.path.exists(job_tracking_dict[job_ind]["job_entire_folder"]):
                    logging.info("Deleting its folder")
                    rmtree(job_tracking_dict[job_ind]["job_entire_folder"])  # delete job folder
                del job_tracking_dict[job_ind]



def assign_tasks_over_available_jobs(current_tasks_path, number_of_jobs_to_send,max_n_tasks_per_job):
    tasks_dict = pickle.load(open(current_tasks_path, "rb"))  # load tasks file
    if len(tasks_dict) == 0:
        return []
    tasks = np.array(list(tasks_dict.keys())[:max_n_tasks_per_job*number_of_jobs_to_send])
    np.random.shuffle(tasks)
    tasks_chunk_keys = np.array_split(tasks, min(number_of_jobs_to_send, len(tasks_dict)))
    tasks_chunks = [{key: tasks_dict[key] for key in key_chunks} for key_chunks in tasks_chunk_keys]
    return tasks_chunks


def current_tasks_pipeline(trimmed_test_msa_path, current_tasks_path, global_results_path, global_results_csv_path, all_jobs_results_folder, total_tasks,
                           args):
    '''

    :param current_run_results_folder:
    :param msa_ind:
    :param msa_path:
    :param args:
    :param global_results_path:
    :param all_jobs_results_folder:
    :param trimmed_test_msa_path:
    :return: Divide tasks between jobs and update results and tasks
    '''
    job_tracking_dict = {}
    job_first_index = 0
    max_n_tasks_per_job = ceil(total_tasks/args.max_n_parallel_jobs)
    logging.info(f"Maximal number of tasks per job is {max_n_tasks_per_job}")
    while len(pickle.load(open(current_tasks_path, "rb"))) > 0 or len(job_tracking_dict)>0:  # Make sure all current tasks are performed
        number_of_available_jobs_to_send = args.max_n_parallel_jobs - len(job_tracking_dict)
        if number_of_available_jobs_to_send > 0:  # Available new jobs.
            logging.debug(f"There are {number_of_available_jobs_to_send} available")
            tasks_per_job = assign_tasks_over_available_jobs(current_tasks_path,
                                                             number_of_available_jobs_to_send,max_n_tasks_per_job)  # Partitioning of tasks over jobs
            for i, job_task in enumerate(tasks_per_job):
                job_ind = job_first_index + i
                logging.info(f"Submitted job number {job_ind}, which will perform {len(job_task)} tasks")
                curr_job_related_files_paths = submit_single_job(all_jobs_results_folder, job_ind, job_task,
                                                                 trimmed_test_msa_path, current_tasks_path, args)
                job_tracking_dict[job_ind] = curr_job_related_files_paths
            number_of_new_job_sent = len(tasks_per_job)
            job_first_index += number_of_new_job_sent
        update_results_tasks_and_jobs(job_tracking_dict, global_results_path, current_tasks_path,global_results_csv_path)
        time.sleep(WAITING_TIME_UPDATE)
    logging.info("Done with the current tasks bunch, deleting all current job folders")



def global_results_to_csv(global_results_dict, csv_path):
    results = [global_results_dict[task_ind].transform_to_dict() for task_ind in global_results_dict]
    df = pd.DataFrame(results)
    df.to_csv(csv_path, sep=CSV_SEP, index=False)


def move_current_tasks_from_pool_to_file(file_paths_path, current_tasks_path, trees_run_directory, args):
    '''

    :param file_paths_path:
    :param current_tasks_path:
    :param trees_run_directory:
    :param args:
    :return: Choose a bunch of MSAs, generate tasks from them, and update the target MSAs list.
    '''
    if os.path.exists(current_tasks_path) and len(pickle.load(
            open(current_tasks_path, "rb"))) > 0:  # if there are currently tasks which are still not performed.
        return
    random.seed(SEED)
    target_msas_list = pickle.load(open(file_paths_path, "rb"))
    current_target_MSAs = target_msas_list[:args.n_MSAs_per_bunch]
    remaining_MSAs = target_msas_list[args.n_MSAs_per_bunch:]
    os.mkdir(trees_run_directory)
    tasks_dict = generate_all_raxml_runs_per_msa(current_target_MSAs, spr_radius_grid_str=args.spr_radius_grid,
                                                 spr_cutoff_grid_str=args.spr_cutoff_grid,
                                                 n_parsimony_tree_objects_per_msa=args.n_raxml_parsimony_trees,
                                                 n_random_tree_objects_per_msa=args.n_raxml_random_trees,
                                                 curr_run_directory=trees_run_directory, seed=SEED)

    logging.info(f"Writing {len(tasks_dict)} tasks belonging to {args.n_MSAs_per_bunch} MSAs to current tasks file : {current_tasks_path}")
    pickle.dump(tasks_dict, open(current_tasks_path, "wb"))
    rmtree(trees_run_directory)
    pickle.dump(tasks_dict, open(current_tasks_path, "wb"))
    pickle.dump(remaining_MSAs, open(file_paths_path, "wb"))
    return len(tasks_dict)


def main():
    parser = main_parser()
    args = parser.parse_args()
    all_jobs_results_folder = generate_results_folder(args.run_prefix)
    all_jobs_general_log_file = os.path.join(all_jobs_results_folder, "log_file.log")
    logging_level = logging.INFO if args.logging_level == "info" else logging.DEBUG
    logging.basicConfig(filename=all_jobs_general_log_file, level=logging_level)
    arguments_path = os.path.join(all_jobs_results_folder, "arguments")
    with open(arguments_path, 'w') as JOB_ARGUMENTS:
        JOB_ARGUMENTS.write(f"Arguments are: {args}")
    logging.info('#Started running')
    global_results_folder = os.path.join(RESULTS_FOLDER, f'global_shared_results_{args.run_prefix}')
    file_paths_path = os.path.join(RESULTS_FOLDER, "global_file_paths")
    global_results_path = os.path.join(global_results_folder, 'global_results_dict')
    trimmed_test_msa_path = os.path.join(global_results_folder, "TEST_MSA")
    global_csv_path = os.path.join(global_results_folder, f'global_csv{CSV_SUFFIX}')
    current_tasks_path = os.path.join(global_results_folder, 'current_tasks')
    trees_run_directory = os.path.join(all_jobs_results_folder, 'starting_trees_generation')
    # extract files
    if not args.use_existing_global_data:
        create_or_clean_dir(global_results_folder)
        target_msas_list = generate_file_path_list_and_test_msa(args, trimmed_test_msa_path)
        logging.info("Generating glboal results and tasks from beggining")
        pickle.dump({}, open(global_results_path, "wb"))
        pickle.dump(target_msas_list, open(file_paths_path, "wb"))
    else:
        logging.info("Using existing global results and tasks")
    target_msas_list = pickle.load(open(file_paths_path, "rb"))
    total_msas_done = 0
    total_msas_overall = len(target_msas_list)
    logging.info(f"Number of target MSAs: {total_msas_overall}, at each iteration {args.n_MSAs_per_bunch} are handled")
    i = 0
    while len(target_msas_list) > 0 : #sanity check
        i += 1
        logging.info(f"iteration {i} starts, time = {time.strftime('%m/%d/%Y, %H:%M:%S', time.localtime())} ")
        total_tasks = move_current_tasks_from_pool_to_file(file_paths_path, current_tasks_path, trees_run_directory, args)
        # Perform pipeline on current MSA, making sure that all tasks in current_tasks_pool are performed.
        current_tasks_pipeline(trimmed_test_msa_path, current_tasks_path, global_results_path, global_csv_path, all_jobs_results_folder, total_tasks,
                               args)
        # Final procedures
        target_msas_list = pickle.load(open(file_paths_path, "rb"))  # Update new tasks.
        logging.debug(f"Size of target MSAs list: {len(target_msas_list)}")

        global_results = pickle.load(open(global_results_path, "rb"))  # Update new results.
        logging.debug(f"Size of Global results: {len(global_results)}")
        total_msas_done += args.n_MSAs_per_bunch
        logging.info(f"iteration {i} done, time = {time.strftime('%m/%d/%Y, %H:%M:%S', time.localtime())} ")
        logging.info(f"So far done with {total_msas_done}/{total_msas_overall} of the MSAs ")


    logging.info(f"Done with all MSAs")


if __name__ == "__main__":
    main()
