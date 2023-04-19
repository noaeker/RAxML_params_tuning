import sys

if sys.platform == "linux" or sys.platform == "linux2":
    PROJECT_ROOT_DIRECRTORY = "/groups/pupko/noaeker/RAxML_params_tuning"
else:
    PROJECT_ROOT_DIRECRTORY = "/Users/noa/Workspace/RAxML_params_tuning"
sys.path.append(PROJECT_ROOT_DIRECRTORY)

from raw_data_generation.msa_runs import generate_all_raxml_runs_per_msa
from side_code.config import *
from side_code.code_submission import submit_linux_job, submit_local_job, generate_argument_list,generate_argument_str, execute_command_and_write_to_log
from side_code.file_handling import create_dir_if_not_exists, create_or_clean_dir, extract_alignment_files_from_dirs, extract_alignment_files_from_general_csv
from side_code.code_submission import is_job_done
from side_code.MSA_manipulation import remove_MSAs_with_not_enough_seq_and_locis, trim_MSA, get_alignment_data
from raw_data_generation.job_runner_side_funcs import main_parser, get_job_related_files_paths
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
import datetime


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
    curr_job_related_details["job_start_time"] = time.time()
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
    if not os.path.exists(trimmed_test_msa_path):
        test_msa_path = file_path_list_full[0]
        trim_MSA(test_msa_path, trimmed_test_msa_path, number_of_sequences=10, max_n_loci=500, loci_shift=0)
        logging.debug("Alignment files are " + str(file_path_list))
        random.seed(SEED)
    if args.remove_existing_msas:
        file_path_list = random.sample(file_path_list_full, min(args.n_MSAs, len(file_path_list_full)))
    else:
        file_path_list = file_path_list_full
    logging.info(
        "Using {} MSAs with at least {} sequences and {} positions".format(len(file_path_list), args.min_n_seq,
                                                                               args.min_n_loci))
    logging.info(
        f"Sampling {len(file_path_list)} random MSAs")
    return file_path_list

def update_tasks_and_results(job_raxml_runs_done_obj,current_results,current_tasks):
    logging.debug(f"Job done size is {len(job_raxml_runs_done_obj)}")
    current_results.update(job_raxml_runs_done_obj)  # update new results
    logging.debug(f"Current results dict size is now {len(current_results)}")
    # update tasks dictionary
    for MSA in job_raxml_runs_done_obj:
        if MSA in current_tasks:
            del current_tasks[MSA]


def terminate_current_job(job_ind, job_tracking_dict):
    logging.info(
        f"Job {job_ind} is done, time = {time.strftime('%m/%d/%Y, %H:%M:%S', time.localtime())}")
    if not LOCAL_RUN:
        pass
        #logging.info(f"Deleting job {job_ind} to make sure it is removed")
        #delete_current_job_cmd = f"qstat | grep {job_tracking_dict[job_ind]['job_name']} | xargs qdel"
        #execute_command_and_write_to_log(delete_current_job_cmd, print_to_log=True)
    if os.path.exists(job_tracking_dict[job_ind]["job_entire_folder"]):
        logging.info(f"Deleting job {job_ind} folder")
        #try:
        #    rmtree(job_tracking_dict[job_ind]["job_entire_folder"])
        #except:
        #    logging.info(f"Could not delete folder {job_tracking_dict[job_ind]['job_entire_folder']}")
            # delete job folder
    del job_tracking_dict[job_ind]
    logging.info(f"job {job_ind} deleted from job tracking dict")




def  check_jobs_status(job_tracking_dict, current_results, current_tasks,timeout, csv_path):
    data_changed_flag = 0
    for job_ind in list(job_tracking_dict.keys()):
        if is_job_done(job_tracking_dict[job_ind]["job_log_folder"], started_file=job_tracking_dict[job_ind]["job_started_file"], job_start_time=job_tracking_dict[job_ind]["job_start_time"], timeout= timeout):
            if os.path.exists(job_tracking_dict[job_ind]["job_local_done_dump"]):
                try:
                    job_raxml_runs_done_obj = pickle.load(open(job_tracking_dict[job_ind]["job_local_done_dump"], "rb"))
                    logging.info(f"Job done size {len(job_raxml_runs_done_obj)}")
                    update_tasks_and_results(job_raxml_runs_done_obj, current_results,
                                             current_tasks)
                    logging.info(f"Current results dict size is {len(current_results)}")
                    logging.info(f"Current tasks dict size is {len(current_tasks)}")
                    logging.info(f"Sanity check: total is {len(current_results)+len(current_tasks)}")
                    terminate_current_job(job_ind, job_tracking_dict)  # fully terminate current job
                    data_changed_flag = 1
                except Exception as e:
                    logging.info(f"Couldn't update file although job is done, e = {e}")

        else: #else, try to update it's results
            if os.path.exists(job_tracking_dict[job_ind]["job_local_done_dump"]) and os.path.getsize(
                    job_tracking_dict[job_ind]["job_local_done_dump"]) > 0:
                try:
                    job_raxml_runs_done_obj = pickle.load(open(job_tracking_dict[job_ind]["job_local_done_dump"], "rb"))
                    update_tasks_and_results(job_raxml_runs_done_obj, current_results,
                                             current_tasks)
                except Exception as e:
                    logging.debug(f"Couldn't update file, e = {e}")
    if data_changed_flag:
        logging.info(f"Writing all results to csv in {csv_path}")
        global_results_to_csv(current_results, csv_path)


    # if job is done, remove it from dictionary







def assign_MSA_tasks_over_available_jobs(current_tasks_per_msa, number_of_jobs_to_send, max_n_MSAs_per_job, prev_running_MSAs): # load tasks file
    n_tasks = len(current_tasks_per_msa)
    if n_tasks == 0:
        return []
    else:
        logging.info(f"Giving {max_n_MSAs_per_job} for each available job")
        tasks_per_job = []
        relevant_MSAs_list=  [msa for msa in current_tasks_per_msa.keys() if msa not in prev_running_MSAs]+[msa for msa in current_tasks_per_msa.keys() if msa in prev_running_MSAs]
        current_running_MSAs = relevant_MSAs_list[:max_n_MSAs_per_job*number_of_jobs_to_send]
        MSAs_ind_per_job = np.array_split(range(len(current_running_MSAs)), min(number_of_jobs_to_send, len(current_tasks_per_msa)))
        for MSAs_group in MSAs_ind_per_job:
            curr_job_tasks = {current_running_MSAs[ind]: current_tasks_per_msa[current_running_MSAs[ind]] for ind in MSAs_group}
            tasks_per_job.append(curr_job_tasks)
        return tasks_per_job, current_running_MSAs



def finish_all_running_jobs(job_tracking_dict):
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


def current_tasks_pipeline(trimmed_test_msa_path, current_tasks, current_results, all_jobs_results_folder, csv_path,
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
    max_n_MSAs_per_job = args.max_n_tasks_per_job if args.max_n_tasks_per_job>0 else ceil(len(current_tasks)/args.max_n_parallel_jobs)
    logging.info(f"Maximal number of MSAs per job is {max_n_MSAs_per_job}")
    run_or_done_MSAs = set()
    while len(current_tasks) > 0:  # Make sure all current tasks are performed
        number_of_available_jobs_to_send = args.max_n_parallel_jobs - len(job_tracking_dict)
        if number_of_available_jobs_to_send > 0:  # Available new jobs.
            logging.info(f"## Currently {len(job_tracking_dict)} jobs are running  ,  {number_of_available_jobs_to_send} available jobs are about to be sent!")
            logging.info(f"Remaining MSAs to be fully done: {len(current_tasks)}")
            tasks_per_job, current_running_MSAs = assign_MSA_tasks_over_available_jobs(current_tasks,
                                                                 number_of_available_jobs_to_send, max_n_MSAs_per_job, run_or_done_MSAs)  # Partitioning of tasks over jobs
            run_or_done_MSAs.update(current_running_MSAs)
            for i, MSAs_tasks in enumerate(tasks_per_job):
                job_ind = job_first_index + i
                logging.info(f"Submitted job number {job_ind}, which will work {len(MSAs_tasks)} MSAs")
                curr_job_related_files_paths = submit_single_job(all_jobs_results_folder, job_ind, MSAs_tasks,
                                                                 trimmed_test_msa_path,args)
                job_tracking_dict[job_ind] = curr_job_related_files_paths
                time.sleep(args.waiting_time_between_job_submissions) # wait 3 seconds between job sendings
            number_of_new_job_sent = len(tasks_per_job)
            job_first_index += number_of_new_job_sent
        check_jobs_status(job_tracking_dict, current_results, current_tasks,timeout= args.timeout, csv_path = csv_path)
        time.sleep(args.waiting_time_between_iterations)
    logging.info("Done with the current msa tasks bunch")
    logging.info(f"Current job_tracking_dict keys are {job_tracking_dict.keys()}" )
    finish_all_running_jobs(job_tracking_dict)
    #time.sleep(15)


def global_results_to_csv(global_results_dict, csv_path):
    results = [global_results_dict[MSA][task_ind].transform_to_dict() for MSA in global_results_dict for task_ind in global_results_dict[MSA]]
    df = pd.DataFrame(results)
    df.to_csv(csv_path, sep=CSV_SEP, index=False)




def update_existing_job_results(directory):
    current_job_data = {}
    logging.info("Update existing tasks from folder")
    for path in [f for f in glob(str(directory)+"/**", recursive=True) if Path(f).is_file() and Path(f).name.startswith('local_raxml_done')]:

        logging.info(f"Updated tasks in {path}")
        job_raxml_runs_done_obj = pickle.load(open(path, "rb"))
        current_job_data.update(job_raxml_runs_done_obj)
        logging.info(f"Updated tasks and results for {path}")
    for path in [Path(f) for f in glob(str(directory)+"/**", recursive=True) if Path(f).is_dir() and Path(f).name.startswith('iter')]:
        try:
            rmtree(path)
            logging.info(f"removed folder {path}")
        except:
            logging.info(f"Could not delete folder {path}")
    return current_job_data

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
    now = datetime.datetime.now()
    all_runs_csv_outputs_folder = os.path.join(RESULTS_FOLDER,'all_final_csvs_results')
    create_dir_if_not_exists(all_runs_csv_outputs_folder)
    specific_shared_results_folder = os.path.join(RESULTS_FOLDER, f'global_shared_results_{run_prefix}')
    file_paths_path = os.path.join(specific_shared_results_folder, f"global_file_paths_{run_prefix}")
    trimmed_test_msa_path = os.path.join(specific_shared_results_folder, "TEST_MSA")
    trees_run_directory = os.path.join(all_jobs_results_folder, 'starting_trees_generation')
    if not args.use_existing_global_data:
        create_dir_if_not_exists(specific_shared_results_folder)
        target_msas_list = generate_file_path_list_and_test_msa(args, trimmed_test_msa_path)
        logging.info("Generating glboal results, file paths and tasks from beggining")
        pickle.dump(target_msas_list, open(file_paths_path, "wb"))
    else:
        logging.info("Using existing global results and tasks")
    date_str = f'{run_prefix}{now.year}_{now.month}_{now.hour}_{now.minute}'
    global_csv_path = os.path.join(all_runs_csv_outputs_folder, f'global_csv_{run_prefix}_{date_str}_{CSV_SUFFIX}')

    csv_files_in_folder = [os.path.join(args.old_msas_folder, f) for f in
                           os.listdir(args.old_msas_folder) if f.endswith(CSV_SUFFIX)]
    dfs_in_folder = []
    for f in csv_files_in_folder:
        try:
            dfs_in_folder.append(pd.read_csv(f, sep=CSV_SEP))
        except:
            pass
    logging.info(f"Combining CSV files: {csv_files_in_folder}")
    if len(dfs_in_folder)>0:
        existing_msas = pd.concat(dfs_in_folder, sort=False)["msa_path"].unique()
    else:
        existing_msas = []

    with open(file_paths_path, "rb") as FILE_PATHS:
        target_msas_list = pickle.load(FILE_PATHS)
    if args.remove_existing_msas:
        logging.info(f"Removing existing msas in {args.old_msas_folder}")
        target_msas_list = [p for p in target_msas_list if p not in existing_msas]
    else:
        logging.info("Using only existing MSAs")
        target_msas_list = [p for p in target_msas_list if p in existing_msas]
    total_msas_done = 0
    total_msas_overall = len(target_msas_list)
    logging.info(f"Number of target MSAs: {total_msas_overall}, at each iteration {args.n_MSAs_per_bunch} are handled")
    i = 0
    current_results = {}
    while i==0 or len(remaining_MSAs ) > 0 or i==0: #sanity check
        i += 1
        logging.info(f"iteration {i} starts, time = {time.strftime('%m/%d/%Y, %H:%M:%S', time.localtime())} ")
        random.seed(SEED)
        if args.n_MSAs_per_bunch>0:
            logging.info("Sampling {args.n_MSAs_per_bunch} from the entire set of MSAs")
            current_target_MSAs = target_msas_list[:args.n_MSAs_per_bunch]
        else:
            current_target_MSAs = target_msas_list
        logging.info(f"Current target MSAs are of length: {len(current_target_MSAs)}")
        create_or_clean_dir(trees_run_directory)


        current_tasks = generate_all_raxml_runs_per_msa(current_target_MSAs, spr_radius_grid_str=args.spr_radius_grid,
                                                        spr_cutoff_grid_str=args.spr_cutoff_grid,
                                                        n_parsimony_tree_objects_per_msa=args.n_raxml_parsimony_trees,
                                                        n_random_tree_objects_per_msa=args.n_raxml_random_trees,
                                                        curr_run_directory=trees_run_directory, seed=SEED, msa_type = args.msa_type, mul = args.mul)

        logging.info(f"Generating tasks belonging to {args.n_MSAs_per_bunch} MSAs ")
        # Perform pipeline on current MSA, making sure that all tasks in current_tasks_pool are performed.
        curr_iterartion_results_folder = os.path.join(all_jobs_results_folder,f"iter_{i}")
        os.mkdir(curr_iterartion_results_folder)
        current_tasks_pipeline(trimmed_test_msa_path, current_tasks, current_results,curr_iterartion_results_folder, csv_path = global_csv_path,
                               args=args) # finishes when tasks are done


        # Final procedures
        logging.info(f"Current results size is {len(current_results)} and will be saved to path: {global_csv_path} ")
        logging.info("Updating all results to csv")
        global_results_to_csv(current_results, global_csv_path)
        remaining_MSAs = [path for path in target_msas_list if path not in current_results.keys()]
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
