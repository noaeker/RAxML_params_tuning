
from side_code.config import *
from side_code.raxml import RF_distances, unify_text_files
from side_code.MSA_manipulation import remove_env_path_prefix
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from pandas.api.types import is_numeric_dtype
from ML_pipeline.ML_config import *

import os





def edit_raw_data_for_ML(data, epsilon):
    #All features
    data["msa_name"] = data["msa_path"].apply(lambda s: remove_env_path_prefix(s))
    data["is_global_max"] = (data["delta_ll_from_overall_msa_best_topology"] <= epsilon).astype('int')
    # data["is_global_max"] = data.groupby('msa_path').transform(lambda x: x<= x.quantile(0.1))["delta_ll_from_overall_msa_best_topology"]
    data["relative_time"] = data["elapsed_running_time"] / data["test_norm_const"]
    mean_default_running_time = \
        data[data["type"] == "default"].groupby('msa_path')['relative_time'].mean().reset_index().rename(
            columns={'relative_time': 'mean_default_time'})
    data["feature_starting_tree_bool"] = data["starting_tree_type"] == "pars"
    data = data.merge(mean_default_running_time, on='msa_path')
    data["normalized_relative_time"] = data["relative_time"] / data["mean_default_time"]
    data["feature_diff_vs_best_tree"] = \
        data[['msa_path',"feature_optimized_ll"]].groupby(['msa_path']).transform(lambda x: ((x - x.max())) / x.max())
    data["feature_brlen_opt_effect"] = data["feature_optimized_ll"] - data["starting_tree_ll"]
    data["feature_seq_to_loci"] = data["feature_n_seq"] / data["feature_n_loci"]
    data["feature_seq_to_unique_loci"] = data["feature_n_seq"] / data["feature_n_unique_sites"]
    non_default_data = data[data["type"] != "default"].copy()
    default_data= data[data["type"] == "default"].copy()
    mean_transformations = non_default_data.groupby('msa_path').transform(lambda vec: np.mean(vec))
    averaged_columns = []
    for col in TREE_FEATURES_LIST:
        name = col+"_averaged_per_entire_MSA"
        averaged_columns.append(name)
        non_default_data[name] = mean_transformations[col]
    #std_transformations = non_default_data.groupby('msa_path').transform(lambda vec :(vec - vec.mean()) / vec.std()+0.01)
    #for col in TREE_FEATURES_LIST:
    #    non_default_data[col] = std_transformations[col]
    return {"non_default": non_default_data, "default": default_data}



def get_average_results_on_default_configurations_per_msa(default_data, n_sample_points, seed
                                                          ):
    default_results = pd.DataFrame()
    for i in range(n_sample_points):
        seed = seed + 1
        sampled_data_parsimony = default_data[default_data["starting_tree_type"] == "pars"].groupby(
            by=["msa_path"]).sample(n=10, random_state=seed)
        sampled_data_random = default_data[default_data["starting_tree_type"] == "rand"].groupby(
            by=["msa_path"]).sample(n=10, random_state=seed)
        sampled_data = pd.concat([sampled_data_parsimony, sampled_data_random])

        run_metrics = sampled_data.groupby(
            by=["msa_path", "best_msa_ll"]).agg(default_final_err = ('delta_ll_from_overall_msa_best_topology', np.min),
                                                default_status = ('is_global_max',np.max), default_total_time = ('normalized_relative_time', np.sum),default_mean_err=('delta_ll_from_overall_msa_best_topology', np.mean),
                                                default_pct_success=('is_global_max', np.mean)).reset_index()
        diff_topologies = sampled_data.groupby(
            by=["msa_path", "best_msa_ll",'tree_clusters_ind']).max().reset_index()
        diff_topologies_run_metrics = diff_topologies.groupby(
            by=["msa_path"]).agg(
                                                default_n_distinct_topologies=(
                                                'tree_clusters_ind', np.count_nonzero)).reset_index().drop(columns = ['msa_path'])

        all_default_metrics = pd.concat([diff_topologies_run_metrics,run_metrics], axis = 1)
        default_results = default_results.append(all_default_metrics )
    return default_results



def get_best_parsimony_config_per_cluster(curr_run_directory, best_configuration_per_starting_tree_pars, normalizing_const, max_dist_options):
    logging.debug(f"Clustering parsimony trees based on distances:{max_dist_options} ")
    parsimony_tree_data = best_configuration_per_starting_tree_pars.sort_values(by='starting_tree_ind')
    parsimony_trees_path = os.path.join(curr_run_directory, 'parsimony_trees')
    unify_text_files(parsimony_tree_data['starting_tree_object'], parsimony_trees_path, str_given=True)
    distances = np.array(RF_distances(curr_run_directory, trees_path_a=parsimony_trees_path, trees_path_b=None, name="RF"))/normalizing_const
    a = np.zeros((len(parsimony_tree_data.index), len(parsimony_tree_data.index)))
    triu = np.triu_indices(len(parsimony_tree_data.index), 1)
    a[triu] = distances
    a = a.T
    a[triu] = a.T[triu]
    parsimony_choice_per_d = {}
    for d in max_dist_options:
        clustering = AgglomerativeClustering(affinity="precomputed", n_clusters= None, distance_threshold=d, linkage='complete').fit(a)
        parsimony_tree_data["cluster"] = clustering.labels_
        parsimony_tree_data["n_actual_parsimony_clusters"] = clustering.n_clusters_
        best_parsimony_configuration_per_cluster = parsimony_tree_data.sort_values(
            ['cluster', 'predicted_failure_probabilities']).groupby(
            ['cluster']).head(1)
        parsimony_choice_per_d[d] = best_parsimony_configuration_per_cluster
    return parsimony_choice_per_d


#
# def knapsack(optional_configurations,max_time_options):
#     v = list(optional_configurations['predicted_time'].apply(lambda x: round(x * 10)))
#     w = list(optional_configurations['predicted_success_probabilities'].apply(lambda x: round(x * 1000)))
#     logging.debug(f"Getting knapsack solution for w = {w}, v= {v}")
#     results_dict = {}
#     for max_time in max_time_options:
#         knapsack_solution_idx = knapsack_solution(w=w,v=v,max_weight=max_time)
#         logging.debug(f"Indexes chosen: {knapsack_solution_idx}")
#         obtained_solution = optional_configurations.iloc[knapsack_solution_idx].copy()
#         results_dict[max_time] = obtained_solution




def get_best_configuration_per_starting_tree(curr_msa_data_per_tree):
    result = []
    for starting_tree_ind in curr_msa_data_per_tree['starting_tree_ind'].unique():
        for starting_tree_type in curr_msa_data_per_tree['starting_tree_type'].unique():
            starting_tree_data = curr_msa_data_per_tree.loc[(curr_msa_data_per_tree.starting_tree_ind==starting_tree_ind)&(curr_msa_data_per_tree.starting_tree_type==starting_tree_type)]
            starting_tree_data = starting_tree_data.sort_values('predicted_time').head(1)
            result.append(starting_tree_data)
    return pd.concat(result)



def get_MSA_clustering_and_threshold_results(curr_run_directory, msa_data,clusters_max_dist_options,max_starting_trees):

    MSA_results = []
    msa_data['predicted_success_probabilities'] = msa_data[
        'predicted_failure_probabilities'].apply(lambda x: 1 - x)
    best_configuration_per_starting_tree = get_best_configuration_per_starting_tree(msa_data)
    clustering_const = 2 * (np.max(msa_data["feature_n_seq"])) - 3
    best_parsimony_configuration_per_cluster_and_size = get_best_parsimony_config_per_cluster(curr_run_directory,
                                                                                              best_configuration_per_starting_tree.loc[
                                                                                                  best_configuration_per_starting_tree.starting_tree_type == "pars"],
                                                                                              normalizing_const=clustering_const,
                                                                              max_dist_options=clusters_max_dist_options)
    for max_dist in best_parsimony_configuration_per_cluster_and_size:
        possible_configurations = pd.concat(
            [best_parsimony_configuration_per_cluster_and_size[max_dist], best_configuration_per_starting_tree.loc[
                best_configuration_per_starting_tree.starting_tree_type == "rand"]]).sort_values('predicted_failure_probabilities')
        for n_trees in range(1,min(max_starting_trees, len(possible_configurations.index))+1):
            curr_method_chosen_starting_trees = possible_configurations.head(n_trees)
            features_columns = MSA_FEATURES_LIST+AVERAGED_FEATURES
            curr_tree_selection_metrics = curr_method_chosen_starting_trees.groupby(["msa_path"]+features_columns).agg(
                feature_total_time_predicted=('predicted_time', np.sum), total_actual_time=('normalized_relative_time', np.sum),
                feature_sum_of_predicted_success_probability=('predicted_success_probabilities', np.sum),
                status=('is_global_max', np.max),
                diff=('delta_ll_from_overall_msa_best_topology', np.min)).reset_index().copy()
            curr_tree_selection_metrics['n_parsimony_clusters'] = len(
                best_parsimony_configuration_per_cluster_and_size[max_dist].index)
            curr_tree_selection_metrics["feature_clusters_max_dist"] = max_dist
            curr_tree_selection_metrics["feature_n_trees_used"] = len(curr_method_chosen_starting_trees.index)
            curr_tree_selection_metrics["feature_n_parsimony_trees_used"] = len(curr_method_chosen_starting_trees.loc[
                                                                            curr_method_chosen_starting_trees.starting_tree_type == "pars"].index)
            curr_tree_selection_metrics["feature_n_random_trees_used"] = len(curr_method_chosen_starting_trees.loc[
                                                                         curr_method_chosen_starting_trees.starting_tree_type == "rand"].index)
            MSA_results.append(curr_tree_selection_metrics)
    return MSA_results



def try_different_tree_selection_metodologies(curr_run_directory, data,clusters_max_dist_options,max_starting_trees):
    logging.info("Trying different procedures to select most promising trees")
    all_msa_validation_performance = []
    msa_paths =  data["msa_path"].unique()
    for msa_path in msa_paths:
        print(msa_path)
        msa_data = data.loc[data.msa_path==msa_path]
        curr_msa_results = get_MSA_clustering_and_threshold_results(curr_run_directory, msa_data,clusters_max_dist_options,max_starting_trees)
        all_msa_validation_performance.extend(curr_msa_results)
    return pd.concat(all_msa_validation_performance)
