import sys

if sys.platform == "linux" or sys.platform == "linux2":
    PROJECT_ROOT_DIRECRTORY = "/groups/pupko/noaeker/RAxML_params_tuning"
else:
    PROJECT_ROOT_DIRECRTORY = "/Users/noa/Workspace/RAxML_params_tuning"
sys.path.append(PROJECT_ROOT_DIRECRTORY)

from side_code.raxml import extract_param_from_raxmlNG_log, raxml_search
from side_code.file_handling import create_or_clean_dir, unify_text_files
#from side_code.MSA_manipulation import get_msa_type
from msa_runs import generate_test_msa_raxml_run
from side_code.basic_trees_manipulation import *
from job_runner_side_funcs import job_parser, get_job_related_files_paths
import pickle
import numpy as np
from shutil import rmtree


def single_tree_RAxML_run(curr_run_directory, single_raxml_run_obj, tmp_starting_tree_path, fake = False):
    '''

    :param curr_run_directory:
    :param msa_path:
    :param starting_tree_path:
    :param path_prefix:
    :param param_obj:
    :param cpus:
    :return: Runs RAxML starting from starting_tree_path based on a grid of parameters
    '''
    run_directory = os.path.join(curr_run_directory, "current_single_tree")
    create_or_clean_dir(run_directory)
    msa_type = single_raxml_run_obj.msa_type
    prefix = "default_params" if single_raxml_run_obj.params_config == {} else "non_default_params"
    curr_param_run_directory = os.path.join(run_directory, prefix)
    create_or_clean_dir(curr_param_run_directory)
    raxml_search_results = raxml_search(curr_param_run_directory, single_raxml_run_obj.msa_path, msa_type, prefix,
                                        single_raxml_run_obj.params_config,
                                        tmp_starting_tree_path, fake = fake)
    rmtree(run_directory)
    return raxml_search_results


def raxml_run_on_test_msa(args, tmp_starting_tree_path):
    '''

    :param args:
    :param tmp_starting_tree_path:
    :return:  runs raxml 20 times on the example MSA
    '''
    test_msa_folder = os.path.join(args.curr_job_folder, "test_msa_results")
    os.mkdir(test_msa_folder)
    test_raxml_run = generate_test_msa_raxml_run(args.test_msa, test_msa_folder, seed=SEED, msa_type = args.test_msa_type)
    with open(tmp_starting_tree_path, 'w') as TMP_STARTING_TREE_PATH:
        TMP_STARTING_TREE_PATH.write(test_raxml_run.starting_tree_object.write(format=1))
    total_test_time = 0
    for i in range(args.n_iters_test):
        logging.debug(f"iter {i} of test")
        curr_i_folder = os.path.join(test_msa_folder, str(i))
        os.mkdir(curr_i_folder)
        test_results = single_tree_RAxML_run(curr_i_folder, test_raxml_run, tmp_starting_tree_path, fake = args.fake_run)
        total_test_time = total_test_time + test_results["elapsed_running_time"]
    rmtree(test_msa_folder)
    return total_test_time


def main():
    parser = job_parser()
    args = parser.parse_args()
    already_started_indication_file = os.path.join(args.curr_job_folder,"started")
    if os.path.exists(already_started_indication_file):
        return
    with open(already_started_indication_file ,"w") as STARTED:
        STARTED.write("STARTED!")
    job_related_file_paths = get_job_related_files_paths(args.curr_job_folder, args.job_ind)
    job_local_tasks_path, job_local_done_dump_path, general_log_path, job_local_stop_running_path = \
        job_related_file_paths[
            "job_local_tasks_path"], \
        job_related_file_paths[
            "job_local_done_dump"], \
        job_related_file_paths[
            "job_log_path"], job_related_file_paths["job_local_stop_running_path"]

    logging_level = logging.INFO if args.logging_level == "info" else logging.DEBUG
    logging.basicConfig(filename=general_log_path, level=logging_level)
    job_arguments_path = os.path.join(args.curr_job_folder, "job_arguments")
    with open(job_arguments_path, 'w') as JOB_ARGUMENTS:
        JOB_ARGUMENTS.write(f"Job arguments are: {args}")
    logging.info(f'#Started running on job {args.job_ind}\n')
    with open(job_local_tasks_path, "rb") as LOCAL_TASKS_PATH:
        job_tasks_dict_per_MSA = pickle.load(LOCAL_TASKS_PATH)
    job_done_dict = {}
    tmp_starting_tree_path = os.path.join(args.curr_job_folder, "tmp_tree")
    for MSA in job_tasks_dict_per_MSA:
        logging.info(f"Working on MSA: {MSA}")
        MSA_tasks = job_tasks_dict_per_MSA[MSA]
        msa_job_done_dict = {}
        first_total_test_time = raxml_run_on_test_msa(args, tmp_starting_tree_path)
        logging.info(f"Total test time1 is: {first_total_test_time}")
        total_test_time = raxml_run_on_test_msa(args, tmp_starting_tree_path)
        logging.info(f"Total test time2 is: {total_test_time}")
        start_time = time.time()
        for i, task_key in (enumerate(MSA_tasks)):
            if os.path.exists(job_local_stop_running_path):  # break out of the loop if all tasks are done
                break
            end_time = time.time() # check currnet time
            if (end_time-start_time)>args.time_between_tests:
                logging.info(f"{args.time_between_tests} second have passed, re-performing the test")
                total_test_time = raxml_run_on_test_msa(args, tmp_starting_tree_path)
                logging.info(f"Total test time is: {total_test_time}")
            logging.info(f"Performing task number {i + 1}/{len(MSA_tasks)}, time = {time.strftime('%m/%d/%Y, %H:%M:%S', time.localtime())}")
            raxml_run = MSA_tasks[task_key]
            with open(tmp_starting_tree_path, 'w') as TMP_STARTING_TREE_PATH:
                TMP_STARTING_TREE_PATH.write(raxml_run.starting_tree_object.write(format=1))

            results = single_tree_RAxML_run(args.curr_job_folder, raxml_run, tmp_starting_tree_path, fake = args.fake_run)
            results["test_norm_const"] = total_test_time
            results["job_ind"] = args.job_ind
            logging.debug(f"Current task results: {results}")
            raxml_run.set_run_results(results)
            msa_job_done_dict[task_key] = raxml_run
        job_done_dict[MSA] = msa_job_done_dict
        with open(job_local_done_dump_path, "wb") as LOCAL_DONE_DUMP:
            pickle.dump(job_done_dict, LOCAL_DONE_DUMP)
        logging.info("Done with current MSA")
    logging.info("Done with current job")


if __name__ == "__main__":
    main()
