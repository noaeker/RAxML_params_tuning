
from side_code.config import *
from side_code.raxml import rf_distance, RF_distances, unify_text_files
from side_code.file_handling import create_dir_if_not_exists
from side_code.MSA_manipulation import remove_env_path_prefix
from ML_pipeline.ML_algorithms_and_hueristics import regressor, classifier
from ML_pipeline.knapsack import knapsack_solution
import pandas as pd
import numpy as np
from shutil import rmtree
from sklearn.cluster import AgglomerativeClustering
from scipy.signal import savgol_filter
from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import random
from sklearn import linear_model

import os

from scipy.cluster.hierarchy import dendrogram, linkage
import argparse


tree_features_list = ["feature_mean_branch_length", "feature_mean_internal_branch_length",
             "feature_tree_MAD", "feature_largest_branch_length","feature_minimal_branch_length","feature_median_branch_length","feature_25_pct_branch_length","feature_largest_distance_between_taxa","feature_smallest_distance_between_taxa","feature_25_pct_distance_between_taxa","feature_75_pct_distance_between_taxa"]

MSA_features_list = ["feature_n_seq","feature_n_loci","feature_msa_type","feature_constant_sites_pct","feature_gap_var","feature_gap_max_by_min","feature_n_unique_sites","feature_frac_unique_sites","feature_pypythia_msa_difficulty"]


def scale_if_needed(vec):
    if is_numeric_dtype(vec):
        if np.var(vec)>0:
            return (vec - vec.mean()) / vec.std()
    else:
        return vec

def mean_if_needed(vec):
    if is_numeric_dtype(vec):
        return np.mean(vec)
    else:
        return vec

def edit_raw_data_for_ML(data, epsilon):
    data["msa_name"] = data["msa_path"].apply(lambda s: remove_env_path_prefix(s))
    data["is_global_max"] = (data["delta_ll_from_overall_msa_best_topology"] <= epsilon).astype('int')
    # data["is_global_max"] = data.groupby('msa_path').transform(lambda x: x<= x.quantile(0.1))["delta_ll_from_overall_msa_best_topology"]
    data["relative_time"] = data["elapsed_running_time"] / data["test_norm_const"]
    mean_default_running_time = \
        data[data["type"] == "default"].groupby('msa_path')['relative_time'].mean().reset_index().rename(
            columns={'relative_time': 'mean_default_time'})
    data["starting_tree_bool"] = data["starting_tree_type"] == "pars"
    data = data.merge(mean_default_running_time, on='msa_path')
    data["normalized_relative_time"] = data["relative_time"] / data["mean_default_time"]
    data["feature_diff_vs_best_tree"] = \
        data[['msa_path',"feature_optimized_ll"]].groupby(['msa_path']).transform(lambda x: ((x - x.max())) / x.max())
    data["feature_brlen_opt_effect"] = data["feature_optimized_ll"] - data["starting_tree_ll"]
    data["feature_seq_to_loci"] = data["feature_n_seq"] / data["feature_n_loci"]
    data["feature_seq_to_unique_loci"] = data["feature_n_seq"] / data["feature_n_unique_sites"]
    non_default_data = data[data["type"] != "default"].copy()
    default_data= data[data["type"] == "default"].copy()
    relevant_columns = [col for col in non_default_data.columns if col.startswith('feature') and is_numeric_dtype(non_default_data[col]) ]
    mean_transformations = non_default_data.groupby('msa_path').transform(lambda vec: mean_if_needed(vec))
    for col in relevant_columns:
        non_default_data[col+"_mean"] = mean_transformations[col]
    std_transformations = non_default_data.groupby('msa_path').transform(lambda vec :scale_if_needed(vec))
    for col in relevant_columns:
        non_default_data[col] = std_transformations[col]
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




def  get_best_configuration_per_starting_tree(curr_msa_data_per_tree):
    result = []
    for starting_tree_ind in curr_msa_data_per_tree['starting_tree_ind'].unique():
        for starting_tree_type in curr_msa_data_per_tree['starting_tree_type'].unique():
            starting_tree_data = curr_msa_data_per_tree.loc[(curr_msa_data_per_tree.starting_tree_ind==starting_tree_ind)&(curr_msa_data_per_tree.starting_tree_type==starting_tree_type)]
            starting_tree_data = starting_tree_data.sort_values('predicted_time').head(1)
            result.append(starting_tree_data)
    return pd.concat(result)



def get_MSA_clustering_and_threshold_results(curr_run_directory, msa_data,clusters_max_dist_options,max_starting_trees):

    MSA_results = []
    msa_columns = [col for col in msa_data.columns if col.endswith('mean')]
    curr_msa_data_per_tree = msa_data[
        ["msa_path", "starting_tree_ind", "starting_tree_object", "spr_radius", "spr_cutoff", "starting_tree_type",
         "predicted_failure_probabilities", "delta_ll_from_overall_msa_best_topology", "tree_clusters_ind",
         "is_global_max", "predicted_time", "normalized_relative_time"] + msa_columns]
    curr_msa_data_per_tree['predicted_success_probabilities'] = curr_msa_data_per_tree[
        'predicted_failure_probabilities'].apply(lambda x: 1 - x)
    best_configuration_per_starting_tree = get_best_configuration_per_starting_tree(curr_msa_data_per_tree)
    clustering_const = 2 * (np.max(curr_msa_data_per_tree["feature_n_seq_mean"])) - 3
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
            curr_method_chosen_starting_trees = possible_configurations .head(n_trees)
            curr_tree_selection_metrics = (curr_method_chosen_starting_trees.groupby(["msa_path"] + msa_columns).agg(
                total_time_predicted=('predicted_time', np.sum), total_actual_time=('normalized_relative_time', np.sum),
                sum_of_predicted_success_probability=('predicted_success_probabilities', np.sum),
                status=('is_global_max', np.max),
                diff=('delta_ll_from_overall_msa_best_topology', np.min)).reset_index()).copy()
            curr_tree_selection_metrics['n_parsimony_clusters'] = len(
                best_parsimony_configuration_per_cluster_and_size[max_dist].index)
            curr_tree_selection_metrics["clusters_max_dist"] = max_dist
            curr_tree_selection_metrics["n_trees_used"] = len(curr_method_chosen_starting_trees.index)
            curr_tree_selection_metrics["n_parsimony_trees_used"] = len(curr_method_chosen_starting_trees.loc[
                                                                            curr_method_chosen_starting_trees.starting_tree_type == "pars"].index)
            curr_tree_selection_metrics["n_random_trees_used"] = len(curr_method_chosen_starting_trees.loc[
                                                                         curr_method_chosen_starting_trees.starting_tree_type == "rand"].index)
            MSA_results.append(curr_tree_selection_metrics)
    return MSA_results



def try_different_tree_selection_metodologies(curr_run_directory, data,clusters_max_dist_options,max_starting_trees):
    logging.info("Trying different procedures to select most promising trees")
    all_msa_validation_performance = []
    msa_paths =  data["msa_path"].unique()
    for msa_path in msa_paths:
        msa_data = data.loc[data.msa_path==msa_path]
        curr_msa_results = get_MSA_clustering_and_threshold_results(curr_run_directory, msa_data,clusters_max_dist_options,max_starting_trees)
        all_msa_validation_performance.extend(curr_msa_results)
    return pd.concat(all_msa_validation_performance)


def choose_best_tree_selection_algorithm(validation_data, file_paths, args):
    '''
    required_accuracy_per_MSA = {}
    best_clusters_per_MSA = {}
    for msa_path in validation_data["msa_path"].unique():
        data = validation_data.loc[(validation_data.msa_path == msa_path)]
        minimal_difference = data["diff"].min()
        clusters_dict = {}
        for clusters_max_dist in validation_data["clusters_max_dist"].unique():
            clusters_data  = data.loc[validation_data.clusters_max_dist == clusters_max_dist].copy().reset_index().sort_values('total_actual_time')
            required_data = clusters_data.loc[(clusters_data["diff"]<=minimal_difference+0.5)].head(1)
            if len(required_data.index)==0:
                continue
            clusters_dict[clusters_max_dist] = {'required_accuracy': required_data['sum_of_predicted_success_probability'].max(), 'required_time': required_data['total_actual_time'].max()}
            print(clusters_dict[clusters_max_dist])
            #plt.plot(clusters_data['sum_of_predicted_success_probability'],clusters_data["diff"])
            #plt.show()
        best_cluster = min(clusters_dict, key = lambda x: clusters_dict[x]['required_time'])
        best_clusters_per_MSA[msa_path] = best_cluster
        required_accuracy_per_MSA[msa_path] = clusters_dict[best_cluster]['required_accuracy']
    validation_data['required_accuracy'] = validation_data['msa_path'].map(required_accuracy_per_MSA)
    validation_data['best_cluster'] = validation_data['msa_path'].map(best_clusters_per_MSA)
    optimal_validation_data = validation_data.loc[(validation_data.required_accuracy==validation_data.sum_of_predicted_success_probability)&(validation_data.best_cluster==validation_data.clusters_max_dist)].drop(['success_threshold'],axis=1)

    '''





    # accuracy_model = regressor(optimal_validation_data[features], optimal_validation_data["required_accuracy"], 1, file_paths["required_accuracy_model_path"],
    #                        args.lightgbm)
    # cluster_model = regressor(optimal_validation_data[features], optimal_validation_data["best_cluster"], 1,
    #                            file_paths["best_cluster_model_path"],
    #                            args.lightgbm)



    #return accuracy_model, cluster_model

    #return curr_aggregated_data,curr_best_success_threshold, curr_best_clusters_max_dist

    # all_msas = list(np.unique(training_data_algorithms_performance_df["msa_path"]))
    # test_size = int(len(all_msas)*0.2)
    # results = {}
    # for i in range(n_cv):
    #     curr_test_msas = random.sample(all_msas,test_size)
    #     curr_train_msas = [msa for msa in all_msas if msa not in curr_test_msas]
    #     curr_train_data = training_data_algorithms_performance_df.loc[training_data_algorithms_performance_df.msa_path.isin(curr_train_msas)]
    #     curr_test_data = training_data_algorithms_performance_df.loc[
    #         training_data_algorithms_performance_df.msa_path.isin(curr_train_msas)]
    #     curr_aggregated_data = curr_train_data.groupby(
    #         ["success_threshold", "clusters_max_dist"]).agg(mean_status=("status", np.mean),
    #                                                             mean_time=('total_actual_time', np.mean),
    #                                                             mean_delta_ll=('diff', np.mean)).reset_index()
    #     curr_best_configuration = curr_aggregated_data[curr_aggregated_data.mean_delta_ll ==curr_aggregated_data.mean_delta_ll.min()]
    #     curr_best_success_threshold = max(curr_best_configuration["tree_selection_method"]) # extract the number
    #     curr_best_clusters_max_dist =  max(curr_best_configuration["clusters_max_dist"]) #ectract the number
    #     best_performance_on_test = np.mean(try_different_tree_selection_metodologies(curr_run_directory, curr_test_data, success_threshold_options = [curr_best_success_threshold],clusters_max_dist_options = curr_best_clusters_max_dist)['mean_delta_ll'])
    #     pass

    #curr_aggregated_data = validation_data.groupby(
    #     ["success_threshold", "clusters_max_dist"]).agg(mean_status=("status", np.mean),
    #                                                     mean_parsimony_clusters = ("n_parsimony_clusters", np.mean),
    #                                                     mean_n_tree_used = ("n_trees_used",np.mean),
    #                                                     mean_n_parsimony_tree_used=("n_parsimony_trees_used", np.mean),
    #                                                     mean_n_random_tree_used=("n_random_trees_used", np.mean),
    #                                                     mean_time=('total_actual_time', np.mean),
    #                                                     mean_delta_ll=('diff', np.mean)).reset_index()
    # curr_best_configuration = curr_aggregated_data[
    #     curr_aggregated_data.mean_delta_ll == curr_aggregated_data.mean_delta_ll.min()]
    # curr_best_success_threshold = max(curr_best_configuration["success_threshold"])  # extract the number
    # curr_best_clusters_max_dist = max(curr_best_configuration["clusters_max_dist"])  # exctract the number