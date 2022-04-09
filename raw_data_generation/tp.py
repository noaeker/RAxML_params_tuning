import sys

if sys.platform=="linux" or sys.platform == "linux2":
    PROJECT_ROOT_DIRECRTORY = "/groups/pupko/noaeker/RAxML_params_tuning"
else:
    PROJECT_ROOT_DIRECRTORY = "/Users/noa/Workspace/RAxML_params_tuning"
sys.path.append(PROJECT_ROOT_DIRECRTORY)

from msa_runs import generate_all_msa_single_raxml_runs
from side_code.config import *
from job_runner_side_funcs import submit_linux_job, submit_local_job, generate_argument_list
from side_code.file_handling import create_dir_if_not_exists, create_or_clean_dir, extract_alignment_files_from_dirs
from side_code.code_submission import is_job_done
from side_code.MSA_manipulation import remove_MSAs_with_not_enough_seq_and_locis, trim_MSA
from job_runner_side_funcs import main_parser, get_job_related_files_paths, get_msa_files
import pickle
import os
import time
from shutil import rmtree
import random
import numpy as np
import pandas as pd
from math import ceil
import sys


def generate_results_folder(curr_run_prefix):
    create_dir_if_not_exists(RESULTS_FOLDER)
    curr_run_prefix = os.path.join(RESULTS_FOLDER, curr_run_prefix)
    create_or_clean_dir(curr_run_prefix)
    return curr_run_prefix


def submit_single_job(all_jobs_results_folder, job_ind, curr_job_tasks_list, test_msa_path, args):
    curr_job_folder = os.path.join(all_jobs_results_folder, "job_" + str(job_ind))
    create_or_clean_dir(curr_job_folder)
    curr_job_related_files_paths = get_job_related_files_paths(curr_job_folder, job_ind)
    curr_job_tasks_path = curr_job_related_files_paths["job_local_tasks_path"]
    pickle.dump(curr_job_tasks_list, open(curr_job_tasks_path, "wb"))
    curr_job_log_path = os.path.join(curr_job_folder, str(job_ind) + "_tmp_log")
    run_command = f' python {MAIN_CODE_PATH} --job_ind {job_ind} --curr_job_folder {curr_job_folder} --test_msa {test_msa_path} '
    job_name = args.jobs_prefix + str(job_ind)
    if not LOCAL_RUN:
        submit_linux_job(job_name, curr_job_folder, curr_job_log_path, run_command, args.n_cpus_per_job, job_ind,
                         queue=args.queue)
    else:
        submit_local_job(MAIN_CODE_PATH,
                         ["--job_ind", str(job_ind), "--curr_job_folder", curr_job_folder, "--test_msa", test_msa_path
                          ] + generate_argument_list(args))
    return curr_job_related_files_paths


def generate_file_path_dict_and_test_msa(args, trimmed_test_msa_path):
    file_path_list = extract_alignment_files_from_dirs(args.general_msa_dir)
    if LOCAL_RUN:
        file_path_list = file_path_list[:100]
    logging.info("There are overall {nMSAs} available MSAs ".format(nMSAs=len(file_path_list)))
    file_path_list_full = remove_MSAs_with_not_enough_seq_and_locis(file_path_list, args.min_n_seq, args.min_n_loci)
    test_msa_path = file_path_list_full[0]
    trim_MSA(test_msa_path, trimmed_test_msa_path, number_of_sequences=10, max_n_loci=500, loci_shift=0)
    logging.debug("Alignment files are " + str(file_path_list))
    random.seed(SEED)
    file_path_list = random.sample(file_path_list_full, args.n_MSAs)
    logging.info(
        "There are {} MSAs with at least {} sequences and {} positions".format(len(file_path_list), args.min_n_seq,
                                                                               args.min_n_loci))
    logging.info(
        f"Sampling {args.n_MSAs} random MSAs")
    return {msa_ind:msa_path for msa_ind,msa_path in enumerate(file_path_list)}


def update_msa_results_and_task_list(job_tracking_dict, msa_files):
    msa_results_dict = pickle.load(open(msa_files["RESULTS"], "rb"))
    msa_tasks_dict = pickle.load(open(msa_files["TASKS"], "rb"))
    total_new_tasks_performed = 0
    for job_ind in list(job_tracking_dict.keys()):
        logging.debug("check if job is done")
        if is_job_done(job_tracking_dict[job_ind]["job_log_folder"]):
            logging.info(f"Job {job_ind} is done")
            job_raxml_runs_done_obj = pickle.load(open(job_tracking_dict[job_ind]["job_local_done_dump"],"rb"))
            total_new_tasks_performed = total_new_tasks_performed + len(job_raxml_runs_done_obj)
            logging.debug(f"Job done size is {len(job_raxml_runs_done_obj)}")
            msa_results_dict.update(job_raxml_runs_done_obj)  # update new results
            msa_tasks_dict = {task_ind : msa_tasks_dict[task_ind] for task_ind in msa_tasks_dict if task_ind not in job_raxml_runs_done_obj}  # insert leftover tasks
            rmtree(job_tracking_dict[job_ind]["job_entire_folder"]) # delete job folder
            del job_tracking_dict[job_ind] #
    pickle.dump(msa_results_dict, open(msa_files["RESULTS"], "wb"))
    pickle.dump(msa_tasks_dict, open(msa_files["TASKS"], "wb"))
    logging.debug(f"Current size of global results after update: {len(msa_results_dict)}")
    return total_new_tasks_performed


def assign_msa_tasks_over_available_jobs(msa_files, number_of_jobs_to_send):
    logging.debug("In assign_msa_tasks_over_jobs")
    msa_tasks_dict = pickle.load(open(msa_files["TASKS"], "rb"))
    if len(msa_tasks_dict)==0:
        return []
    msa_tasks_chunk_keys = np.array_split(np.array(list(msa_tasks_dict.keys())),min(number_of_jobs_to_send,len(msa_tasks_dict)))
    tasks_chunks = [{key: msa_tasks_dict[key] for key in key_chunks} for key_chunks in msa_tasks_chunk_keys]
    logging.debug(f"Overall keys to be performed: {len(msa_tasks_dict)}:\n {msa_tasks_dict}")
    logging.debug(f"length of msa tasks dict: {len(msa_tasks_dict)}:\n {msa_tasks_dict}")
    remaining_msa_tasks_dict = {key: msa_tasks_dict[key] for key in msa_tasks_dict.keys() if key not in msa_tasks_dict.keys()}
    pickle.dump(remaining_msa_tasks_dict,open(msa_files["TASKS"], "wb")) #save remaining tasks
    return  tasks_chunks



def distribute_tasks_to_available_jobs(msa_files,total_performed_tasks,job_first_index,all_jobs_results_folder,trimmed_test_msa_path, args,job_tracking_dict,number_of_tasks_per_msa):
    number_of_available_jobs_to_send = args.max_n_parallel_jobs - len(job_tracking_dict)
    if number_of_available_jobs_to_send==0:
        return 0
    tasks_per_job = assign_msa_tasks_over_available_jobs(msa_files,
                                                         number_of_available_jobs_to_send)

    if len(tasks_per_job) > 0:
        t = time.localtime()
        current_time = time.strftime("%m/%d/%Y, %H:%M:%S", t)
        logging.info(f"Current time: {current_time}; number of new available jobs to run: {len(tasks_per_job)}; Perfomed overall {total_performed_tasks} tasks out of {number_of_tasks_per_msa} of current MSA")
    for i, job_task in enumerate(tasks_per_job):
        job_ind = job_first_index + i
        logging.info(f"Submitted job number {job_ind}, which performs {len(job_task)} tasks")
        curr_job_related_files_paths = submit_single_job(all_jobs_results_folder, job_ind, job_task,
                                                         trimmed_test_msa_path, args)
        job_tracking_dict[job_ind] = curr_job_related_files_paths
    return len(tasks_per_job)




def single_msa_pipeline(msa_files,msa_results_folder,msa_path, args,
                            trimmed_test_msa_path,current_running_jobs_folder):
    '''

    :param current_run_results_folder:
    :param msa_ind:
    :param msa_path:
    :param args:
    :param global_results_path:
    :param all_jobs_results_folder:
    :param trimmed_test_msa_path:
    :return: Full MSA pipeline: including job managing
    '''
    msa_results_dict_ALL = {}
    pickle.dump(msa_results_dict_ALL, open(msa_files["RESULTS"], "wb"))
    trees_run_directory = os.path.join(msa_results_folder, 'starting_trees_generation')
    os.mkdir(trees_run_directory)
    msa_tasks_dict = generate_all_msa_single_raxml_runs(msa_path, spr_radius_grid_str=args.spr_radius_grid,
                                                        spr_cutoff_grid_str=args.spr_cutoff_grid,
                                                        n_parsimony_tree_objects_per_msa=args.n_raxml_parsimony_trees,
                                                        n_random_tree_objects_per_msa=args.n_raxml_random_trees,
                                                        curr_run_directory=trees_run_directory, seed=SEED)
    pickle.dump(msa_tasks_dict, open(msa_files["TASKS"], "wb"))
    rmtree(trees_run_directory)
    number_of_tasks_per_msa = len(msa_tasks_dict)
    logging.info(f"Current msa path: {msa_path}, number of tasks to be performed: {number_of_tasks_per_msa}")
    job_tracking_dict = {}
    job_first_index = 0
    total_performed_tasks = 0
    while total_performed_tasks < number_of_tasks_per_msa:
        number_of_new_tasks_sent = distribute_tasks_to_available_jobs(msa_files,total_performed_tasks,job_first_index,current_running_jobs_folder,trimmed_test_msa_path, args,job_tracking_dict,number_of_tasks_per_msa)
        if number_of_new_tasks_sent>0:
            job_first_index += number_of_new_tasks_sent
            new_tasks_performed = update_msa_results_and_task_list(job_tracking_dict, msa_files)
            total_performed_tasks += new_tasks_performed
        time.sleep(WAITING_TIME_UPDATE)


def global_results_to_csv(global_results_dict, csv_path):
    results = [global_results_dict[msa_path][task_ind].transform_to_dict() for msa_path in global_results_dict for task_ind in global_results_dict[msa_path]]
    df = pd.DataFrame(results)
    df.to_csv(csv_path,sep = CSV_SEP, index = False)



def main():
    parser = main_parser()
    args = parser.parse_args()
    all_jobs_results_folder = generate_results_folder(args.run_prefix)
    all_jobs_general_log_file = os.path.join(all_jobs_results_folder, "log_file.log")
    logging_level = logging.INFO if args.logging_level=="info" else logging.DEBUG
    logging.basicConfig(filename=all_jobs_general_log_file, level=logging_level)
    arguments_path = os.path.join(all_jobs_results_folder, "arguments")
    with open(arguments_path, 'w') as JOB_ARGUMENTS:
        JOB_ARGUMENTS.write(f"Arguments are: {args}")
    logging.info('#Started running')
    global_results_folder = os.path.join(RESULTS_FOLDER, 'global_shared_results')
    global_results_path = os.path.join(global_results_folder, 'global_results_dict')
    trimmed_test_msa_path = os.path.join(global_results_folder, "TEST_MSA")
    global_csv_path = os.path.join(global_results_folder, f'global_csv_path{CSV_SUFFIX}')
    file_paths = os.path.join(global_results_folder, "file_paths")
    if not args.use_existing_global_data:
        create_or_clean_dir(global_results_folder)
        global_results = {}
        pickle.dump(global_results, open(global_results_path, "wb"))
        new_file_path_dict = generate_file_path_dict_and_test_msa(args, trimmed_test_msa_path)
        pickle.dump(new_file_path_dict, open(file_paths, "wb"))


    target_MSAs_dict = pickle.load(open(file_paths, "rb"))
    total_msas_todo = len(target_MSAs_dict)
    total_msas_done =0
    while len(target_MSAs_dict)>0:
        random.seed(SEED)
        msa_ind = random.choice(list(target_MSAs_dict.keys())) # choose a random MSA
        msa_path = target_MSAs_dict[msa_ind]
        msa_results_folder = os.path.join(all_jobs_results_folder, f'msa_{msa_ind}')
        create_or_clean_dir(msa_results_folder)
        current_running_jobs_folder = os.path.join(msa_results_folder, "current_running_jobs")
        os.mkdir(current_running_jobs_folder)
        msa_files = get_msa_files(msa_results_folder)
        single_msa_pipeline(msa_files,msa_results_folder,msa_path, args,
                            trimmed_test_msa_path,current_running_jobs_folder)
        total_msas_done += 1
        logging.info(f"Done with MSA {msa_path} ,  so far done with {total_msas_done}/{total_msas_todo} of the MSAs ")
        msa_results = pickle.load(open(msa_files["RESULTS"], "rb"))
        global_results = pickle.load(open(global_results_path, "rb"))
        global_results[msa_path] = msa_results  # Update global results with new MSA
        pickle.dump(global_results, open(global_results_path, "wb"))
        target_MSAs_dict = pickle.load(open(file_paths, "rb"))
        del target_MSAs_dict[msa_ind]  # remove # Remove MSA from file path list
        pickle.dump(target_MSAs_dict, open(file_paths, "wb"))
        global_results_to_csv(global_results, global_csv_path) # write results to csv file



    logging.info(f"Done with all MSAs")


if __name__ == "__main__":
    main()
