import sys

if sys.platform == "linux" or sys.platform == "linux2":
    PROJECT_ROOT_DIRECRTORY = "/groups/pupko/noaeker/RAxML_params_tuning"
else:
    PROJECT_ROOT_DIRECRTORY = "/Users/noa/Workspace/RAxML_params_tuning"
sys.path.append(PROJECT_ROOT_DIRECRTORY)

from msa_runs import generate_all_raxml_runs_per_msa
from side_code.config import *
from side_code.code_submission import submit_linux_job, submit_local_job, generate_argument_list,generate_argument_str, execute_command_and_write_to_log
from side_code.file_handling import create_dir_if_not_exists, create_or_clean_dir, extract_alignment_files_from_dirs, extract_alignment_files_from_general_csv
from side_code.code_submission import is_job_done
from side_code.MSA_manipulation import remove_MSAs_with_not_enough_seq_and_locis, trim_MSA, get_alignment_data
from job_runner_side_funcs import main_parser, get_job_related_files_paths
from pathlib import Path
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
from pathlib import Path
from glob import glob


def generate_results_folder(curr_run_prefix):
    create_dir_if_not_exists(RESULTS_FOLDER)
    curr_run_prefix = os.path.join(RESULTS_FOLDER, curr_run_prefix)
    create_or_clean_dir(curr_run_prefix)
    return curr_run_prefix


def submit_single_job(all_jobs_results_folder, job_ind, curr_job_tasks_list, test_msa_path,args):
    curr_job_folder = os.path.join(all_jobs_results_folder, "job_" + str(job_ind))
    create_or_clean_dir(curr_job_folder)
    curr_job_related_details = get_job_related_files_paths(curr_job_folder, job_ind)
    curr_job_tasks_path = curr_job_related_details["job_local_tasks_path"]
    pickle.dump(curr_job_tasks_list, open(curr_job_tasks_path, "wb"))
    curr_job_log_path = os.path.join(curr_job_folder, str(job_ind) + "_tmp_log")
    run_command = f' python {MAIN_CODE_PATH} --job_ind {job_ind} --curr_job_folder {curr_job_folder} --test_msa {test_msa_path}  {generate_argument_str(args)} '
    job_name = f"{job_ind}_{args.jobs_prefix}"
    if not LOCAL_RUN:
        submit_linux_job(job_name, curr_job_folder, curr_job_log_path, run_command, args.n_cpus_per_job, job_ind,
                         queue=args.queue)
    else:
        submit_local_job(MAIN_CODE_PATH,
                         ["--job_ind", str(job_ind), "--curr_job_folder", curr_job_folder, "--test_msa", test_msa_path,
                          ] + generate_argument_list(args))
    curr_job_related_details["job_start_time"] = time.localtime()
    curr_job_related_details["job_name"] = job_name
    return curr_job_related_details


def generate_file_path_list_and_test_msa(args, trimmed_test_msa_path):
    if args.use_files_from_csv:
        file_path_list = extract_alignment_files_from_general_csv(os.path.join(CSV_MSAs_FOLDER,"sampled_datasets.csv"))
    else:
        file_path_list = extract_alignment_files_from_dirs(args.general_msa_dir)
        if args.MSAs_pool_size > 0:
            file_path_list = file_path_list[:args.MSAs_pool_size]
    logging.info("There are overall {nMSAs} available MSAs ".format(nMSAs=len(file_path_list)))
    file_path_list_full = remove_MSAs_with_not_enough_seq_and_locis(file_path_list, args.min_n_seq, args.max_n_seq, args.min_n_loci)
    test_msa_path = file_path_list_full[0]
    trim_MSA(test_msa_path, trimmed_test_msa_path, number_of_sequences=10, max_n_loci=500, loci_shift=0)
    logging.debug("Alignment files are " + str(file_path_list))
    random.seed(SEED)
    file_path_list = random.sample(file_path_list_full, min(args.n_MSAs, len(file_path_list_full)))
    logging.info(
        "Using {} MSAs with at least {} sequences and {} positions".format(len(file_path_list), args.min_n_seq,
                                                                               args.min_n_loci))
    logging.info(
        f"Sampling {len(file_path_list)} random MSAs")
    return file_path_list

def update_tasks_and_results(job_raxml_runs_done_obj,current_results,current_tasks):
    logging.info(f"Job done size is {len(job_raxml_runs_done_obj)}")
    current_results.update(job_raxml_runs_done_obj)  # update new results
    logging.info(f"Current results dict size is now {len(current_results)}")
    # update tasks dictionary
    for task_ind in job_raxml_runs_done_obj:
        del current_tasks[task_ind]


def  check_jobs_status(job_tracking_dict, current_results, current_tasks,timeout, update_anyway = False):
    for job_ind in list(job_tracking_dict.keys()):
        if is_job_done(job_tracking_dict[job_ind]["job_log_folder"], started_file=job_tracking_dict[job_ind]["job_started_file"], job_start_time=job_tracking_dict[job_ind]["job_start_time"], timeout= timeout) or update_anyway:  # if job is done, remove it from dictionary
            logging.info(
                f"Job {job_ind} is done, time = {time.strftime('%m/%d/%Y, %H:%M:%S', time.localtime())}")
            if not LOCAL_RUN:
                logging.info(f"Deleting job {job_ind} to make sure it is removed")
                delete_current_job_cmd = f"qstat | grep {job_tracking_dict[job_ind]['job_name']} | xargs qdel"
                execute_command_and_write_to_log(delete_current_job_cmd, print_to_log= True)
            if os.path.exists(job_tracking_dict[job_ind]["job_local_done_dump"]) and os.path.getsize(
                    job_tracking_dict[job_ind]["job_local_done_dump"]) > 0:
                logging.info("RAxML runs done object exists and will be updated to current tasks and results!")
                job_raxml_runs_done_obj = pickle.load(open(job_tracking_dict[job_ind]["job_local_done_dump"], "rb"))
                update_tasks_and_results(job_raxml_runs_done_obj, current_results,
                                         current_tasks)
            if os.path.exists(job_tracking_dict[job_ind]["job_entire_folder"]):
                logging.info(f"Deleting job {job_ind} folder")
                try:
                    rmtree(job_tracking_dict[job_ind]["job_entire_folder"])
                except:
                    logging.info(f"Could not delete folder {job_tracking_dict[job_ind]['job_entire_folder']}")
                            # delete job folder
            del job_tracking_dict[job_ind]
            logging.info(f"job {job_ind} deleted from job tracking dict")






def assign_tasks_over_available_jobs(current_tasks, number_of_jobs_to_send,max_n_tasks_per_job): # load tasks file
    if len(current_tasks) == 0:
        return []
    tasks = np.array(list(current_tasks.keys())[:max_n_tasks_per_job*number_of_jobs_to_send])
    np.random.shuffle(tasks)
    tasks_chunk_keys = np.array_split(tasks, min(number_of_jobs_to_send, len(current_tasks)))
    tasks_chunks = [{key: current_tasks[key] for key in key_chunks} for key_chunks in tasks_chunk_keys]
    return tasks_chunks


def current_tasks_pipeline(trimmed_test_msa_path, current_tasks, current_results, all_jobs_results_folder,
                           args):
    '''
    '''
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
    max_n_tasks_per_job = ceil(len(current_tasks)/args.max_n_parallel_jobs)
    logging.info(f"Maximal number of tasks per job is {max_n_tasks_per_job}")
    while len(current_tasks) > 0:  # Make sure all current tasks are performed
        number_of_available_jobs_to_send = args.max_n_parallel_jobs - len(job_tracking_dict)
        if number_of_available_jobs_to_send > 0:  # Available new jobs.
            logging.info(f"## Currently {len(job_tracking_dict)} jobs are running  ,  {number_of_available_jobs_to_send} available jobs are about to be sent!")
            tasks_per_job = assign_tasks_over_available_jobs(current_tasks,
                                                             number_of_available_jobs_to_send,max_n_tasks_per_job)  # Partitioning of tasks over jobs
            for i, job_task in enumerate(tasks_per_job):
                job_ind = job_first_index + i
                logging.info(f"Submitted job number {job_ind}, which will perform {len(job_task)} tasks")
                curr_job_related_files_paths = submit_single_job(all_jobs_results_folder, job_ind, job_task,
                                                                 trimmed_test_msa_path,args)
                job_tracking_dict[job_ind] = curr_job_related_files_paths
            number_of_new_job_sent = len(tasks_per_job)
            job_first_index += number_of_new_job_sent
        check_jobs_status(job_tracking_dict, current_results, current_tasks,timeout= args.timeout)
        time.sleep(WAITING_TIME_UPDATE)
    logging.info("Done with the current tasks bunch")
    logging.info(f"Current job_tracking_dict keys are {job_tracking_dict.keys()}" )
    for job_ind in list(job_tracking_dict.keys()): # remove all remaining folders
        logging.info(f"Deleting job {job_ind} to make sure it is removed")
        if not LOCAL_RUN:
            delete_current_job_cmd = f"qstat | grep {job_tracking_dict[job_ind]['job_name']} | xargs qdel"
            execute_command_and_write_to_log(delete_current_job_cmd, print_to_log=True)
        try:
            if os.path.exists(job_tracking_dict[job_ind]["job_entire_folder"]):
                logging.info(f"Deleting {job_ind} folder since all tasks are done")
                rmtree(job_tracking_dict[job_ind]["job_entire_folder"])
                del job_tracking_dict[job_ind]
        except Exception:
            logging.info(f"Couldn't delete folder {job_tracking_dict[job_ind]['job_entire_folder']}")
            pass
        #time.sleep(15)






def global_results_to_csv(global_results_dict, csv_path):
    results = [global_results_dict[task_ind].transform_to_dict() for task_ind in global_results_dict]
    df = pd.DataFrame(results)
    df.to_csv(csv_path, sep=CSV_SEP, index=False)



def move_current_results_to_global_results(current_results_dict, global_results_path,global_results_csv_path):

    logging.info("Moving current results to global results")
    global_results_dict = pickle.load(open(global_results_path, "rb"))
    logging.info(f"Global results size is {len(global_results_dict)}")
    logging.info(f"Current results size is {len(current_results_dict)} and will be added to global dict ")
    global_results_dict.update(current_results_dict)  # update new results
    global_results_to_csv(global_results_dict, global_results_csv_path)
    pickle.dump(global_results_dict, open(global_results_path, "wb"))




# def update_existing_job_results(directory, global_results_path, current_tasks_path):
#     current_jobs
#     logging.info("Update existing tasks from folder")
#     for path in [f for f in glob(str(directory)+"/**", recursive=True) if Path(f).is_file() and Path(f).name.startswith('local_raxml_done')]:
#
#         logging.info(f"Updated tasks in {path}")
#         job_raxml_runs_done_obj = pickle.load(open(path, "rb"))
#         update_tasks_and_results(job_raxml_runs_done_obj, global_results_path,
#                                  current_tasks_path)
#         logging.info(f"Updated tasks and resutls for {path}")
#     for path in [Path(f) for f in glob(str(directory)+"/**", recursive=True) if Path(f).is_dir() and Path(f).name.startswith('iter')]:
#         try:
#             rmtree(path)
#             logging.info(f"removed folder {path}")
#         except:
#             logging.info(f"Could not delete folder {path}")

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
    with open(all_jobs_general_log_file, 'w'):  # empty log file
        pass
    logging.info('#Started running')
    run_prefix = args.existing_global_data_to_use if args.existing_global_data_to_use is True else args.run_prefix
    global_results_folder = os.path.join(RESULTS_FOLDER, f'global_shared_results_{run_prefix}')
    global_results_path = os.path.join(global_results_folder, 'global_results_dict')
    file_paths_path = os.path.join(global_results_folder, f"global_file_paths_{run_prefix}")
    trimmed_test_msa_path = os.path.join(global_results_folder, "TEST_MSA")
    global_csv_path = os.path.join(global_results_folder, f'global_csv{CSV_SUFFIX}')
    trees_run_directory = os.path.join(all_jobs_results_folder, 'starting_trees_generation')
    #update_existing_tasks(all_jobs_results_folder, global_results_path, global_csv_path,
    #                      current_tasks_path)
    # extract files
    if not args.use_existing_global_data or not os.path.exists(global_results_folder):
        create_or_clean_dir(global_results_folder)
        target_msas_list = generate_file_path_list_and_test_msa(args, trimmed_test_msa_path)
        logging.info("Generating glboal results, file paths and tasks from beggining")
        pickle.dump({}, open(global_results_path, "wb"))
        pickle.dump(target_msas_list, open(file_paths_path, "wb"))
    else:
        logging.info("Using existing global results and tasks")

    target_msas_list = pickle.load(open(file_paths_path, "rb"))
    total_msas_done = 0
    total_msas_overall = len(target_msas_list)
    logging.info(f"Number of target MSAs: {total_msas_overall}, at each iteration {args.n_MSAs_per_bunch} are handled")
    i = 0
    while len(target_msas_list) > 0 or i==0: #sanity check
        i += 1
        logging.info(f"iteration {i} starts, time = {time.strftime('%m/%d/%Y, %H:%M:%S', time.localtime())} ")
        random.seed(SEED)
        current_target_MSAs = target_msas_list[:args.n_MSAs_per_bunch]
        remaining_MSAs = target_msas_list[args.n_MSAs_per_bunch:]
        create_or_clean_dir(trees_run_directory)
        current_results = {}
        current_tasks = generate_all_raxml_runs_per_msa(current_target_MSAs, spr_radius_grid_str=args.spr_radius_grid,
                                                        spr_cutoff_grid_str=args.spr_cutoff_grid,
                                                        n_parsimony_tree_objects_per_msa=args.n_raxml_parsimony_trees,
                                                        n_random_tree_objects_per_msa=args.n_raxml_random_trees,
                                                        curr_run_directory=trees_run_directory, seed=SEED)


        logging.info(f"Generating overall {len(current_tasks)} tasks belonging to {args.n_MSAs_per_bunch} MSAs ")
        # Perform pipeline on current MSA, making sure that all tasks in current_tasks_pool are performed.
        curr_iterartion_results_folder = os.path.join(all_jobs_results_folder,f"iter_{i}")
        os.mkdir(curr_iterartion_results_folder)
        current_tasks_pipeline(trimmed_test_msa_path, current_tasks, current_results,curr_iterartion_results_folder,
                               args) # finishes when tasks are done


        # Final procedures
        move_current_results_to_global_results(current_results, global_results_path, global_csv_path)
        target_msas_list = remaining_MSAs
        pickle.dump(target_msas_list , open(file_paths_path, "wb")) # Done with current filess
        total_msas_done += args.n_MSAs_per_bunch
        #with open(all_jobs_general_log_file,'w'): #empty log file
        #    pass
        logging.info(f"###### iteration {i} done, time = {time.strftime('%m/%d/%Y, %H:%M:%S', time.localtime())} ")
        logging.info(f"###### So far done with {total_msas_done}/{total_msas_overall} of the MSAs ")
        try:

            rmtree(curr_iterartion_results_folder)
        except:
            logging.info(f"Could not delete folder {curr_iterartion_results_folder}")

    logging.info(f"Done with all MSAs")


if __name__ == "__main__":
    main()
