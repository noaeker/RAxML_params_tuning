import sys

if sys.platform=="linux" or sys.platform == "linux2":
    PROJECT_ROOT_DIRECRTORY = "/groups/pupko/noaeker/RAxML_params_tuning"
else:
    PROJECT_ROOT_DIRECRTORY = "/Users/noa/Workspace/RAxML_params_tuning"
sys.path.append(PROJECT_ROOT_DIRECRTORY)

from side_code.raxml import extract_param_from_raxmlNG_log, raxml_search
from side_code.file_handling import create_or_clean_dir, unify_text_files
from side_code.MSA_manipulation import get_msa_type
from msa_runs import generate_test_msa_raxml_run
from side_code.basic_trees_manipulation import *
from job_runner_side_funcs import job_parser, get_job_related_files_paths
import pickle
import numpy as np
from shutil import rmtree


def calculate_rf_dist(rf_file_path, curr_run_directory, prefix="rf"):
    rf_prefix = os.path.join(curr_run_directory, prefix)
    rf_command = (
        "{raxml_exe_path} --force msa --force perf_threads --rfdist --tree {rf_file_path} --prefix {prefix}").format(
        raxml_exe_path=RAXML_NG_EXE, rf_file_path=rf_file_path, prefix=rf_prefix)
    execute_command_and_write_to_log(rf_command)
    rf_log_file_path = rf_prefix + ".raxml.log"
    relative_rf_dist = extract_param_from_raxmlNG_log(rf_log_file_path, "rf_dist")
    return relative_rf_dist


def rf_distance(curr_run_directory, tree_str_a, tree_str_b):
    rf_folder = os.path.join(curr_run_directory, f"rf_calculations")
    create_or_clean_dir(rf_folder)
    rf_output_path = os.path.join(rf_folder, "rf_calculations")
    rf_first_phase_trees = unify_text_files([tree_str_a, tree_str_b], rf_output_path, str_given=True)
    rf = calculate_rf_dist(rf_first_phase_trees, rf_folder,
                           prefix="rf_calculations")
    return rf


def process_spefic_starting_tree_search_RAxML_runs(curr_run_directory, given_tree_search_data):
    '''

    :param curr_run_directory:
    :param given_tree_search_data:
    :return:
    '''
    best_tree_search_ll = max(given_tree_search_data["final_ll"])
    given_tree_search_data["curr_starting_tree_best_ll"] = best_tree_search_ll
    best_tree_search_topology = max(
        given_tree_search_data[given_tree_search_data["final_ll"] == best_tree_search_ll]['final_tree_topology'])
    given_tree_search_data["rf_from_curr_starting_tree_best_topology"] = given_tree_search_data[
        "final_tree_topology"].apply(
        lambda x: rf_distance(curr_run_directory, x, best_tree_search_topology))
    given_tree_search_data["delta_ll_from_curr_starting_tree_best_topology"] = np.where(
        (given_tree_search_data["rf_from_curr_starting_tree_best_topology"]) > 0,
        best_tree_search_ll - given_tree_search_data["final_ll"], 0)
    return given_tree_search_data


def process_all_msa_RAxML_runs(curr_run_directory, given_msa_data):
    '''

    :param curr_run_directory:
    :param given_msa_data:
    :return:
    '''
    best_msa_ll = max(given_msa_data["final_ll"])
    given_msa_data["best_msa_ll"] = best_msa_ll
    best_msa_tree_topology = max(given_msa_data[given_msa_data["final_ll"] == best_msa_ll]['final_tree_topology'])
    given_msa_data["rf_from_overall_msa_best_topology"] = given_msa_data["final_tree_topology"].apply(
        lambda x: rf_distance(curr_run_directory, x, best_msa_tree_topology))
    given_msa_data["delta_ll_from_overall_msa_best_topology"] = np.where(
        (given_msa_data["rf_from_overall_msa_best_topology"]) > 0, best_msa_ll - given_msa_data["final_ll"], 0)
    return given_msa_data


def single_tree_RAxML_run(curr_run_directory, single_raxml_run_obj, tmp_starting_tree_path):
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
    msa_type = get_msa_type(single_raxml_run_obj.msa_path)
    prefix = "default_params" if single_raxml_run_obj.params_config=={} else "non_default_params"
    curr_param_run_directory = os.path.join(run_directory, prefix)
    create_or_clean_dir(curr_param_run_directory)
    raxml_search_results = raxml_search(curr_param_run_directory, single_raxml_run_obj.msa_path, msa_type, prefix, single_raxml_run_obj.params_config,
                                        tmp_starting_tree_path)
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
    test_raxml_run = generate_test_msa_raxml_run(args.test_msa, test_msa_folder, seed=SEED)
    with open(tmp_starting_tree_path, 'w') as TMP_STARTING_TREE_PATH:
        TMP_STARTING_TREE_PATH.write(test_raxml_run.starting_tree_object.write(format=1))
    total_test_time = 0
    for i in range(args.n_iters_test):
        test_results = single_tree_RAxML_run(test_msa_folder, test_raxml_run, tmp_starting_tree_path)
        create_or_clean_dir(test_msa_folder)
        total_test_time = total_test_time + test_results["elapsed_running_time"]
    rmtree(test_msa_folder)
    return total_test_time



def main():
    parser = job_parser()
    args = parser.parse_args()
    job_related_file_paths = get_job_related_files_paths(args.curr_job_folder, args.job_ind)
    job_local_tasks_path, job_local_done_dump_path, general_log_path, = \
        job_related_file_paths[
            "job_local_tasks_path"], \
        job_related_file_paths[
            "job_local_done_dump"], \
        job_related_file_paths[
            "job_log_path"]

    logging_level = logging.INFO if args.logging_level == "info" else logging.DEBUG
    logging.basicConfig(filename=general_log_path, level=logging_level)
    job_arguments_path = os.path.join(args.curr_job_folder, "job_arguments")
    with open(job_arguments_path,'w') as JOB_ARGUMENTS:
        JOB_ARGUMENTS.write(f"Job arguments are: {args}")
    logging.info(f'#Started running on job {args.job_ind}\n')
    job_tasks_dict = pickle.load(open(job_local_tasks_path,"rb"))
    job_done_dict = {}
    tmp_starting_tree_path = os.path.join(args.curr_job_folder, "tmp_tree")
    total_test_time = raxml_run_on_test_msa(args, tmp_starting_tree_path)
    for i,task_ind in (enumerate(job_tasks_dict)):
        logging.info(f"Performing task number {i + 1}/{len(job_tasks_dict)}")
        raxml_run = job_tasks_dict[task_ind]
        with open(tmp_starting_tree_path, 'w') as TMP_STARTING_TREE_PATH:
            TMP_STARTING_TREE_PATH.write(raxml_run.starting_tree_object.write(format=1))

        results = single_tree_RAxML_run(args.curr_job_folder, raxml_run, tmp_starting_tree_path)
        results["test_norm_const"] = total_test_time
        logging.debug(f"Current task results: {results}")
        raxml_run.set_run_results(results)
        job_done_dict[task_ind] = raxml_run
        pickle.dump(job_done_dict, open(job_local_done_dump_path, "wb"))
    logging.info("Done with current job")





if __name__ == "__main__":
    main()
