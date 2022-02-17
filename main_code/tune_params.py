from raxml import *
from help_functions import *
import numpy as np
import pandas as pd


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


def single_tree_RAxML_run(curr_run_directory, msa_path, starting_tree_path, path_prefix, param_obj, msa_stats, cpus):
    '''

    :param curr_run_directory:
    :param msa_path:
    :param starting_tree_path:
    :param path_prefix:
    :param param_obj:
    :param cpus:
    :return: Runs RAxML starting from starting_tree_path based on a grid of parameters
    '''
    tree_results = pd.DataFrame()
    path_run_directory = os.path.join(curr_run_directory, path_prefix)
    create_or_clean_dir(path_run_directory)
    default_prefix = f"params_default"
    default_param_run_directory = os.path.join(path_run_directory, default_prefix)
    create_or_clean_dir(default_param_run_directory)
    default_raxml_search_results = raxml_search(msa_stats,default_param_run_directory, msa_path, default_prefix, {},
                                                cpus=cpus, starting_tree_path=starting_tree_path)
    default_raxml_search_results["run_name"] = "default"
    default_raxml_search_results.update(msa_stats)
    tree_results = tree_results.append(default_raxml_search_results, ignore_index=True)
    for j, params_config in enumerate(param_obj):
        prefix = f"params_{j}"
        curr_param_run_directory = os.path.join(path_run_directory, prefix)
        create_or_clean_dir(curr_param_run_directory)
        curr_raxml_search_results = raxml_search(msa_stats,curr_param_run_directory, msa_path, prefix, params_config,
                                                 cpus=cpus, starting_tree_path=starting_tree_path)
        curr_raxml_search_results["run_name"] = prefix
        curr_raxml_search_results.update(msa_stats)
        if msa_stats["remove_output_files"]:
            shutil.rmtree(curr_param_run_directory)
        tree_results = tree_results.append(curr_raxml_search_results, ignore_index=True)
    if msa_stats["remove_output_files"]:
        shutil.rmtree(path_run_directory)

    return tree_results


def run_raxml_on_several_spcific_tree_type(curr_run_directory, msa_path, msa_stats, param_obj, tree_type, n):
    '''

    :param curr_run_directory:
    :param msa_path:
    :param msa_stats:
    :param param_obj:
    :param tree_type:
    :return: The function generates the required number of trees of a given type and runs RAxML on each of them
    '''
    all_given_tree_type_results = pd.DataFrame()
    parsimony_topologies_path, elapsed_time_p = generate_n_unique_tree_topologies_as_starting_trees(
        n=n,
        original_file_path=msa_path,
        curr_run_directory=curr_run_directory,
        curr_msa_stats=msa_stats, seed=SEED,
        tree_type=tree_type)
    parsimony_objects = generate_multiple_tree_object_from_newick(parsimony_topologies_path)
    logging.info("Running RAxML on parsimony trees")
    starting_tree_path = os.path.join(curr_run_directory, f"curr_{tree_type}_starting_tree_path.tree")
    for i, starting_tree_obj in enumerate(parsimony_objects):
        with open(starting_tree_path, 'w') as STARTING_PATH:
            STARTING_PATH.write(starting_tree_obj.write(format=1))
        single_tree_results = single_tree_RAxML_run(curr_run_directory, msa_path, starting_tree_path,
                                                    path_prefix=f"{tree_type}_{i}", param_obj=param_obj,
                                                    msa_stats=msa_stats,
                                                    cpus=msa_stats["n_cpus_raxml"])
        processed_single_tree_results = process_spefic_starting_tree_search_RAxML_runs(curr_run_directory, single_tree_results)
        processed_single_tree_results["starting_tree_ind"] = i
        all_given_tree_type_results = pd.concat([all_given_tree_type_results, processed_single_tree_results],
                                                sort=False)
        logging.debug(f"single_tree_results {i}")
    return all_given_tree_type_results


def RAxML_runs_on_given_msa(msa_stats, msa_path, curr_run_directory, param_obj):
    '''

    :param msa_stats:
    :param msa_path:
    :param curr_run_directory:
    :param param_obj:
    :return:  The function runs RAxML on random and parsimony trees for each configuration in param_obj (including the default configuraiton)
    '''
    logging.info("About to run RAxML on parsimony trees")
    parsimony_trees_results = run_raxml_on_several_spcific_tree_type(curr_run_directory, msa_path, msa_stats, param_obj,
                                                                     "pars", n= msa_stats["n_raxml_parsimony_trees"])
    parsimony_df = pd.DataFrame(parsimony_trees_results)
    parsimony_df["tree_type"] = "parsimony"
    logging.info("About to run RAxML on random trees")
    random_trees_results = run_raxml_on_several_spcific_tree_type(curr_run_directory, msa_path, msa_stats, param_obj,
                                                                  "rand", n = msa_stats["n_raxml_random_trees"])
    random_df = pd.DataFrame(random_trees_results)
    random_df["tree_type"] = "random"
    all_tree_runs = pd.concat([parsimony_df, random_df], sort=False)
    processed_msa_runs = process_all_msa_RAxML_runs(curr_run_directory, all_tree_runs)

    return processed_msa_runs


def generate_msa_stats(original_alignment_path, args):
    logging.info("Generating general MSA stats")
    msa_name = get_msa_name(original_alignment_path,args.general_msa_dir)
    curr_msa_folder = os.path.join(args.curr_job_folder, msa_name)
    create_or_clean_dir(curr_msa_folder)
    original_alignment_data = get_alignment_data(
        original_alignment_path)  # list(SeqIO.parse(original, file_type_biopython))
    alignment_df = alignment_list_to_df(original_alignment_data)
    n_seq, n_loci = alignment_df.shape
    msa_path = original_alignment_path
    msa_path_no_extension = os.path.splitext(msa_path)[0]
    if re.search('\w+D[\da-z]+',msa_path_no_extension.split(os.sep)[-2]) is not None:
        msa_type = "DNA"
    else:
        msa_type = "AA"
    msa_stats = {"msa_name": msa_name, "msa_path": msa_path,
                 "original_alignment_path": original_alignment_path,
                 "n_loci": n_loci, "n_seq": n_seq, "msa_folder": curr_msa_folder, "msa_type": msa_type}
    msa_stats.update(vars(args))
    logging.info(f"Succesfully obtained MSA stats: {msa_stats}")
    return msa_stats


def MSA_search_params_tuning_analysis(msa_stats):
    '''

    :param msa_stats:
    :return: The function generates a grid of parameters based on the user's input, runs RAxML on different starting trees based on
    different values of the grid
    '''

    param_grid_str = {"spr_radius": msa_stats["spr_radius_grid"], "spr_cutoff": msa_stats["spr_cutoff_grid"]}
    param_obj = get_param_obj(param_grid_str)
    curr_msa_RAxML_directory = os.path.join(msa_stats["msa_folder"], "RAxML_runs")
    create_or_clean_dir(curr_msa_RAxML_directory)
    raw_msa_RAxML_results = RAxML_runs_on_given_msa(msa_stats, msa_stats["msa_path"], curr_msa_RAxML_directory,
                                                    param_obj)
    return raw_msa_RAxML_results


def main():
    parser = job_parser()
    args = parser.parse_args()
    job_related_file_paths = get_job_related_files_paths(args.curr_job_folder, args.job_ind)
    job_msa_paths_file, general_log_path, job_csv_path, curr_job_status_file = \
        job_related_file_paths[
            "job_msa_paths_file"], \
        job_related_file_paths[
            "general_log_path"], \
        job_related_file_paths[
            "job_csv_path"], \
        job_related_file_paths[
            "job_status_file"]
    with open(job_msa_paths_file, "r") as paths_file:
        curr_job_file_path_list = paths_file.read().splitlines()
    logging.basicConfig(filename=general_log_path, level=LOGGING_LEVEL)
    logging.info(f'#Started running on job {args.job_ind}\nJob arguments are: {args}')

    for file_ind, original_alignment_path in enumerate(curr_job_file_path_list):
        logging.info(f"file ind = {file_ind} original_alignment_path= {original_alignment_path}")
        msa_stats = generate_msa_stats(original_alignment_path, args)
        curr_msa_data_analysis = MSA_search_params_tuning_analysis(msa_stats)[COLUMNS_TO_INCLUDE_CSV]
        curr_msa_data_analysis.to_csv(job_csv_path,mode='a',header = file_ind==0,sep=CSV_SEP)
        if args.remove_output_files:
            shutil.rmtree(msa_stats["msa_folder"])
        open(general_log_path, 'w').close()

    with open(curr_job_status_file, 'w') as job_status_f:
        job_status_f.write("Done")
    logging.info("Current job is Done!")




if __name__ == "__main__":
    main()
