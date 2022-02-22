from side_code.raxml import *
from side_code.help_functions import *
from side_code.config import *
from side_code.basic_trees_manipulation import *
import pickle


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
    given_tree_search_data["rf_from_curr_starting_tree_best_topology"] = given_tree_search_data["final_tree_topology"].apply(
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


def single_tree_RAxML_run(curr_run_directory, msa_path, starting_tree_path, param_config, default = False):
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
    msa_type = get_msa_type(msa_path)
    if default:
        default_prefix = "default_params"
        default_param_run_directory = os.path.join(run_directory, default_prefix)
        create_or_clean_dir(default_param_run_directory)
        #curr_run_directory, msa_path, msa_type, prefix,params_config, starting_tree_path
        raxml_search_results = raxml_search(default_param_run_directory, msa_path, msa_type, default_prefix, {},
                                                    starting_tree_path)

    else:
        prefix = "non_default_params"
        curr_param_run_directory = os.path.join(run_directory, prefix)
        create_or_clean_dir(curr_param_run_directory)
        raxml_search_results = raxml_search(curr_param_run_directory, msa_path, msa_type, prefix, param_config,
                                                 starting_tree_path)
    return  raxml_search_results




def main():
    parser = job_parser()
    args = parser.parse_args()
    job_related_file_paths = get_sampling_job_related_files_paths(args.curr_job_folder, args.job_ind)
    job_local_raxml_runs_path, general_log_path,  = \
        job_related_file_paths[
            "job_local_raxml_runs_path"], \
        job_related_file_paths[
            "general_log_path"], \
        job_related_file_paths[
            "job_csv_path"], \
        job_related_file_paths[
            "job_status_file"]

    logging.basicConfig(filename=general_log_path, level=LOGGING_LEVEL)
    logging.info(f'#Started running on job {args.job_ind}\nJob arguments are: {args}')
    raxml_runs_dict = pickle.load(args.job_raxml_runs_path)
    tmp_starting_tree_path=  os.path.join(args.curr_job_folder, "tmp_tree_path")
    for raxml_ind in raxml_runs_dict:
         with open(tmp_starting_tree_path,'w'):
             tmp_starting_tree_path.write(get_tree_string(raxml_run.starting_tree_object))
         results = single_tree_RAxML_run(args.curr_job_folder, raxml_run.msa_path, tmp_starting_tree_path,raxml_run.params_config)
         raxml_run.set_run_results(results)
         pickle.dump(raxml_runs_dict, open(job_local_raxml_runs_path, "wb"))

    with open(curr_job_status_file, 'w') as job_status_f:
        job_status_f.write("Done")
    logging.info("Current job is Done!")






if __name__ == "__main__":
    main()
