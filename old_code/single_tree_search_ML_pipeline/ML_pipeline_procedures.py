from side_code.MSA_manipulation import remove_env_path_prefix
import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype


def scale_if_needed(vec):
    if is_numeric_dtype(vec):
        if np.var(vec)>0:
            return (vec - vec.mean()) / vec.std()
    else:
        return vec



def edit_raw_data_for_ML(data, epsilon):
    #All features
    data["msa_name"] = data["msa_path"].apply(lambda s: remove_env_path_prefix(s))
    data["is_global_max"] = (data["delta_ll_from_overall_msa_best_topology"] <= epsilon).astype('int')
    # data["is_global_max"] = data.groupby('msa_path').transform(lambda x: x<= x.quantile(0.1))["delta_ll_from_overall_msa_best_topology"]
    data["relative_time"] = data["elapsed_running_time"] / data["test_norm_const"]
    # mean_default_running_time = \
    #     data[data["type"] == "default"].groupby('msa_path')['relative_time'].mean().reset_index().rename(
    #         columns={'relative_time': 'mean_default_time'})
    default_config_per_tree = data[data["type"] == "default"][['msa_path','starting_tree_type','starting_tree_ind','spr_radius','spr_cutoff', 'delta_ll_from_overall_msa_best_topology']].drop_duplicates().reset_index().rename(
            columns={'spr_radius': 'spr_radius_default','spr_cutoff': 'spr_cutoff_default', 'delta_ll_from_overall_msa_best_topology': 'default_delta_ll'})
    non_default_data = data[data["type"] != "default"].copy()
    non_default_data = non_default_data.merge(default_config_per_tree, on = ['msa_path','starting_tree_type','starting_tree_ind'])
    non_default_data["equal_to_default_config"] = (non_default_data["spr_cutoff"] ==non_default_data["spr_cutoff_default"]) & (non_default_data["spr_radius"]==non_default_data["spr_radius_default"])
    non_default_data["relative_time"] = non_default_data["elapsed_running_time"] / non_default_data["test_norm_const"]
    mean_default_by_params_running_time = \
        non_default_data[non_default_data["equal_to_default_config"]].groupby('msa_path')['relative_time'].mean().reset_index().rename(
            columns={'relative_time': 'mean_default_time'})
    non_default_data = non_default_data.merge(mean_default_by_params_running_time, on='msa_path')
    non_default_data["normalized_relative_time"] = non_default_data["relative_time"] / non_default_data["mean_default_time"]
    non_default_data["starting_tree_bool"] = non_default_data["starting_tree_type"] == "pars"
    #data["feature_diff_vs_best_tree"] = \
    #    data[['msa_path',"feature_tree_optimized_ll"]].groupby(['msa_path']).transform(lambda x: ((x - x.max())) / x.max())
    #data["feature_brlen_opt_effect"] = data["feature_tree_optimized_ll"] - data["starting_tree_ll"]
    #non_default_data["feature_seq_to_loci"] = non_default_data["feature_msa_n_seq"] / non_default_data["feature_msa_n_loci"]
    non_default_data["feature_seq_to_unique_loci"] = non_default_data["feature_msa_n_seq"] / non_default_data["feature_msa_n_unique_sites"]
 #   non_default_data["found_best_score"] = non_default_data.groupby("msa_path")['is_global_max'].transform(np.max)
    default_by_params = non_default_data[non_default_data["equal_to_default_config"]].copy()
    numerical_columns = [col for col in non_default_data.columns if
                         col.startswith('feature') and is_numeric_dtype(non_default_data[col])]
    changing_features = non_default_data[numerical_columns + ['msa_path']].groupby('msa_path').apply(
         lambda x: np.var(x)).sum(axis=0)
    starting_tree_level_columns =changing_features[changing_features > 0.001].index
    # #
    starting_tree_level_columns = [col for col in starting_tree_level_columns if col not in ('feature_cluster_size','feature_cluster_size_random ') ]
    MSA_level_columns = [col for col in numerical_columns if col not in starting_tree_level_columns]
    mean_transformations = non_default_data.groupby('msa_path').transform(lambda vec: np.mean(vec))
    averaged_cols = []
    for col in starting_tree_level_columns:
       if not 'PCA' in col and not 'TSNE' in col:
           name = col + "_averaged_per_entire_MSA"
           non_default_data[col + "_averaged_per_entire_MSA"] = mean_transformations[col]
           averaged_cols.append(name)
    std_transformations = non_default_data.groupby('msa_path').transform(lambda vec: vec - (vec - vec.mean()) / vec.std())#(vec - vec.mean()) / vec.std() #(vec - vec.min()) / (vec.max()-vec.min())
    for col in starting_tree_level_columns:
        #if not 'PCA' in col:
            non_default_data[col] =std_transformations[col]
    #non_default_data = non_default_data.drop(columns = starting_tree_level_columns)
    return {"non_default": non_default_data, "default_by_params": default_by_params,"MSA_level_columns": MSA_level_columns,"averaged_MSA_level_columns":averaged_cols  }




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




def get_average_results_on_default_configurations_per_msa(default_data, n_sample_points, seed, n_trees
                                                          ):
    default_results = pd.DataFrame()
    for i in range(n_sample_points):
        seed = seed + 1
        sampled_data = default_data.sample(n=n_trees, random_state=seed)

        run_metrics = sampled_data.groupby(
            by=["msa_path", "best_msa_ll"]).agg(default_final_err = ('delta_ll_from_overall_msa_best_topology', np.min),
                                                default_status = ('is_global_max',np.max), default_total_time = ('normalized_relative_time', np.sum),
                                                ).reset_index()

        diff_topologies = sampled_data.groupby(
            by=["msa_path", "best_msa_ll",'tree_clusters_ind']).max().reset_index()
        diff_topologies_run_metrics = diff_topologies.groupby(
            by=["msa_path"]).agg(
                                                default_n_distinct_topologies=(
                                                'tree_clusters_ind', np.count_nonzero)).reset_index().drop(columns = ['msa_path'])

        all_default_metrics = pd.concat([diff_topologies_run_metrics,run_metrics], axis = 1)
        default_results = default_results.append(all_default_metrics )
    return default_results


#
# def get_best_parsimony_config_per_cluster(curr_run_directory, best_configuration_per_starting_tree_pars, normalizing_const, max_dist_options):
#     logging.debug(f"Clustering parsimony trees based on distances:{max_dist_options} ")
#     parsimony_tree_data = best_configuration_per_starting_tree_pars.sort_values(by='starting_tree_ind')
#     parsimony_trees_path = os.path.join(curr_run_directory, 'parsimony_trees')
#     unify_text_files(parsimony_tree_data['starting_tree_object'], parsimony_trees_path, str_given=True)
#     distances = np.array(RF_distances(curr_run_directory, trees_path_a=parsimony_trees_path, trees_path_b=None, name="RF"))/normalizing_const
#     a = np.zeros((len(parsimony_tree_data.index), len(parsimony_tree_data.index)))
#     triu = np.triu_indices(len(parsimony_tree_data.index), 1)
#     a[triu] = distances
#     a = a.T
#     a[triu] = a.T[triu]
#     parsimony_choice_per_d = {}
#     for d in max_dist_options:
#         clustering = AgglomerativeClustering(affinity="precomputed", n_clusters= None, distance_threshold=d, linkage='complete').fit(a)
#         parsimony_tree_data["cluster"] = clustering.labels_
#         parsimony_tree_data["n_actual_parsimony_clusters"] = clustering.n_clusters_
#         best_parsimony_configuration_per_cluster = parsimony_tree_data.sort_values(
#             ['cluster', 'predicted_uncalibrated_failure_probabilities']).groupby(
#             ['cluster']).head(1)
#         parsimony_choice_per_d[d] = best_parsimony_configuration_per_cluster
#     return parsimony_choice_per_d
#
#
# #
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
    return (curr_msa_data_per_tree.sort_values('predicted_time').groupby(['starting_tree_ind']).head(1))















