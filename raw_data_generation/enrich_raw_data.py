
from side_code.raxml import extract_param_from_raxmlNG_log
from side_code.file_handling import create_or_clean_dir, unify_text_files
import os
from side_code.code_submission import execute_command_and_write_to_log
from side_code.config import *
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
        (given_msa_data["rf_from_overall_msa_best_topology"]) > 0,  given_msa_data["final_ll"]-best_msa_ll, 0)
    return given_msa_data


def main():
    raw_data_path = f''
    raw_data_df = pd.read_csv(raw_data_path)

if __name__ == "__main__":
    main()