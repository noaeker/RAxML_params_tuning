
from raxml import *
import numpy as np
from sklearn.model_selection import ParameterGrid
from generate_training_data import *
from generate_SPR import *


def calculate_rf_dist(rf_file_path, curr_run_directory, prefix="rf"):
    rf_prefix = os.path.join(curr_run_directory, prefix)
    rf_command = (
        "{raxml_exe_path} --force msa --force perf_threads --rfdist --tree {rf_file_path} --prefix {prefix}").format(
        raxml_exe_path=RAXML_NG_EXE, rf_file_path=rf_file_path, prefix=rf_prefix)
    execute_commnand_and_write_to_log(rf_command, run_locally=True)
    rf_log_file_path = rf_prefix + ".raxml.log"
    relative_rf_dist = extract_param_from_raxmlNG_log(rf_log_file_path, "rf_dist")
    return relative_rf_dist


def rf_distance(curr_run_directory, tree_path_a, tree_path_b):
    rf_folder = os.path.join(curr_run_directory, f"rf_calculations")
    create_or_clean_dir(rf_folder)
    rf_output_path = os.path.join(rf_folder, "rf_calculations")
    rf_first_phase_trees = unify_text_files([tree_path_a, tree_path_b], rf_output_path)
    rf = calculate_rf_dist(rf_first_phase_trees, rf_folder,
                           prefix="rf_calculations")
    return rf


def analyze_RAxML_given_msa_runs(curr_run_directory, given_msa_data):
    best_ll = max(given_msa_data["best_ll"])
    best_tree_topology = max(given_msa_data[given_msa_data["best_ll"] == best_ll]['best_tree_topology_path'])
    default_tree_ll = max(given_msa_data[given_msa_data["run_name"] == "default"]['best_ll'])
    default_tree_topology = max(given_msa_data[given_msa_data["run_name"] == "default"]['best_tree_topology_path'])
    given_msa_data["rf_from_best_topology"] = given_msa_data["best_tree_topology_path"].apply(
        lambda x: rf_distance(curr_run_directory, x, best_tree_topology))
    given_msa_data["rf_from_default_topology"] = given_msa_data["best_tree_topology_path"].apply(
        lambda x: rf_distance(curr_run_directory, x, default_tree_topology))
    given_msa_data["delta_ll_from_best_topology"] = given_msa_data["best_ll"] - best_ll
    given_msa_data["delta_ll_from_default_topology"] = given_msa_data["best_ll"] - default_tree_ll
    return given_msa_data


def RAxML_runs_on_given_msa(msa_stats,msa_path, curr_run_directory, param_obj):
    all_msa_runs = pd.DataFrame()
    default_prefix = f"params_default"
    default_param_run_directory = os.path.join(curr_run_directory, default_prefix)
    create_or_clean_dir(default_param_run_directory)
    default_raxml_search_results = raxml_search(default_param_run_directory, msa_path, default_prefix, {},
                                                cpus=1)
    default_raxml_search_results["run_name"] = "default"
    all_msa_runs = all_msa_runs.append(default_raxml_search_results, ignore_index=True)
    for i, params_config in enumerate(param_obj):
        prefix = f"params_{i}"
        curr_param_run_directory = os.path.join(curr_run_directory, prefix)
        create_or_clean_dir(curr_param_run_directory)
        curr_raxml_search_results = raxml_search(curr_param_run_directory, msa_path, prefix, params_config,
                                                 cpus=1)
        curr_raxml_search_results["run_name"] = str(i)
        curr_raxml_search_results.update(msa_stats)
        #delete_dir_content(curr_param_run_directory)
        all_msa_runs = all_msa_runs.append(curr_raxml_search_results, ignore_index=True)
    return all_msa_runs



def extract_msa_features(n, alpha, original_file_path, curr_run_directory,
                                                   curr_msa_stats, seed):
    random_tree_path, elapsed_running_time =generate_n_random_tree_topology_constant_brlen(n, alpha, original_file_path, curr_run_directory,
                                                   curr_msa_stats, seed)
    ll_n_random_trees = raxml_optimize_trees_for_given_msa(original_file_path, "evaluating ll of random trees", random_tree_path, curr_msa_stats,
                                       curr_run_directory, opt_brlen=True, weights=None, return_trees_file=False,
                                       n_cpus=1)






def RAxML_on_MSA(msa_stats, msa_path, curr_run_directory, param_obj):
    logging.info(f" Running RAxML on msa in: {msa_path} ")
    curr_msa_run_directory = os.path.join(curr_run_directory, "RAxML_runs")
    create_or_clean_dir(curr_msa_run_directory)
    curr_msa_RAxML_runs = RAxML_runs_on_given_msa(msa_stats,msa_stats["trimmed_msa_path"], curr_msa_run_directory, param_obj)
    curr_msa_data_analysis = analyze_RAxML_given_msa_runs(curr_run_directory, curr_msa_RAxML_runs)
    return curr_msa_data_analysis





def str_to_linspace(str):
    linespace_nums = [int(n) for n in str.split("_")]
    return np.linspace(linespace_nums[0],linespace_nums[1],linespace_nums[2])


def get_param_obj(n_parsimony_grid_str, n_random_grid_str, spr_radius_grid_str):
    param_grid = {}
    for param_name, param_grid_str in zip(['n_parsimony','n_random','spr_radius'], [n_parsimony_grid_str, n_random_grid_str, spr_radius_grid_str]):
        if param_grid_str!="default":
            linspace = str_to_linspace(param_grid_str)
            param_grid[param_name] = linspace
    param_obj = (ParameterGrid(param_grid))
    return param_obj


def main():
    parser = job_parser()
    args = parser.parse_args()
    job_related_file_paths = get_job_related_files_paths(args.curr_job_folder, args.job_ind)
    job_msa_paths_file, general_log_path, job_csv_path, job_best_csv_path, curr_job_status_file = \
    job_related_file_paths[
        "job_msa_paths_file"], \
    job_related_file_paths[
        "general_log_path"], \
    job_related_file_paths[
        "job_csv_path"], \
    job_related_file_paths["job_only_best_csv_path"], \
    job_related_file_paths[
        "job_status_file"]
    with open(job_msa_paths_file, "r") as paths_file:
        curr_job_file_path_list = paths_file.read().splitlines()
    logging.basicConfig(filename=general_log_path, level=LOGGING_LEVEL)
    logging.info('#Started running on job' + str(args.job_ind))
    logging.info("Job arguments : {}".format(args))

    job_results = pd.DataFrame(
    )
    job_results.to_csv(job_csv_path, index=False)


    job_best_results = pd.DataFrame(
    )
    job_best_results.to_csv(job_best_csv_path, index=False)
    for file_ind, original_alignment_path in enumerate(curr_job_file_path_list):
        msa_name = original_alignment_path.replace(MSAs_FOLDER, "").replace("ref_msa.aa.phy", "").replace(os.path.sep,
                                                                                                          "_")
        logging.info(
            f'#running on file name {msa_name} and ind (relativ to job) {file_ind}  original path= {original_alignment_path}')
        curr_msa_folder = os.path.join(args.curr_job_folder, msa_name)
        create_or_clean_dir(curr_msa_folder)
        msa_stats = handle_msa(curr_msa_folder, original_alignment_path, args.n_seq, args.n_loci)
        msa_stats.update(vars(args))
        extract_raxml_statistics_from_msa(original_alignment_path, f"msa_{file_ind}", msa_stats,  curr_msa_folder)
        logging.info(f"Basic MSA stats {msa_stats}\n")
        param_obj = get_param_obj(args.n_parsimony_grid, args.n_random_grid, args.spr_radius_grid)
        if msa_stats["use_raxml_search"]:
            curr_msa_data_analysis = RAxML_on_MSA(msa_stats, original_alignment_path, curr_msa_folder, param_obj)
        else:
            curr_msa_data_analysis = SPR_on_MSA(msa_stats, original_alignment_path, curr_msa_folder, param_obj)
        job_results = job_results.append(curr_msa_data_analysis, ignore_index=True)
        best_raxml_result = (curr_msa_data_analysis[ curr_msa_data_analysis["rf_from_best_topology"] == 0]).sort_values(
            'elapsed_running_time', ascending=True).head(1)
        job_best_results = job_best_results.append(best_raxml_result)
        job_results.to_csv(job_csv_path)
        job_best_results.to_csv(job_best_csv_path)

    with open(curr_job_status_file, 'w') as job_status_f:
        job_status_f.write("Done")
    logging.info("Current job is done")



if __name__ == "__main__":
    main()
