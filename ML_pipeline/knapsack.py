

import pandas as pd
import numpy as np
from ML_pipeline.ML_pipeline_procedures import get_best_configuration_per_starting_tree,get_best_parsimony_config_per_cluster


def knapsack_on_test_set(curr_run_directory,data_dict,time_model, error_model):
    MSA_results = []
    enriched_test_data = data_dict["full_test_data"]

    time_model_test_data = time_model['selector'].transform(data_dict["X_test"])
    enriched_test_data["predicted_time"] = time_model['best_model'].predict(time_model_test_data)
    error_model_test_data = error_model['selector'].transform(data_dict["X_test"])
    enriched_test_data["predicted_failure_probabilities"] = error_model['calibrated_model'].predict_proba(error_model_test_data)[:,
                                                            0]
    enriched_test_data["predicted_success_probabilities"] = enriched_test_data["predicted_failure_probabilities"].apply(lambda x: 1-x)
    for msa in enriched_test_data["msa_path"].unique():
        curr_msa_data = enriched_test_data[enriched_test_data["msa_path"]==msa]
        fastest_per_starting_tree = get_best_configuration_per_starting_tree(curr_msa_data)
        clustering_const = 2 * (np.max(curr_msa_data["feature_msa_n_seq"])) - 3
        parsimony_configuration = get_best_parsimony_config_per_cluster(curr_run_directory,
                                                                        fastest_per_starting_tree.loc[
                                                                            fastest_per_starting_tree.starting_tree_type == "pars"],
                                                                                                  normalizing_const=clustering_const,
                                                                                                  max_dist_options=[0.5])[0.5]
        random_configuration =  fastest_per_starting_tree.loc[
            fastest_per_starting_tree.starting_tree_type == "rand"]
        fastest_per_starting_tree = pd.concat([parsimony_configuration,random_configuration])
        fastest_per_starting_tree["failure_prob"] = np.log(fastest_per_starting_tree["predicted_failure_probabilities"])
        fastest_per_starting_tree["failure_score"] = (fastest_per_starting_tree["failure_prob"]/enriched_test_data["predicted_time"])*-1
        fastest_per_starting_tree = fastest_per_starting_tree.sort_values(by="failure_score", ascending= False )
        fastest_per_starting_tree["cum_failure_prob"] = fastest_per_starting_tree["failure_prob"].cumsum().apply(lambda x: np.exp(x))
        chosen_trees =   fastest_per_starting_tree[ fastest_per_starting_tree["cum_failure_prob"]>=0.05]
        curr_tree_selection_metrics = (
            chosen_trees.groupby(["msa_path"]).agg(
                total_time_predicted=('predicted_time', np.sum), total_actual_time=('normalized_relative_time', np.sum),
                sum_of_predicted_success_probability=('predicted_success_probabilities', np.sum),
                status=('is_global_max', np.max),
                diff=('delta_ll_from_overall_msa_best_topology', np.min)).reset_index()).copy()

        curr_tree_selection_metrics["n_trees_used"] = len(chosen_trees.index)
        curr_tree_selection_metrics["n_parsimony_trees_used"] = len(chosen_trees.loc[
                                                                        chosen_trees.starting_tree_type == "pars"].index)
        curr_tree_selection_metrics["n_random_trees_used"] = len(chosen_trees.loc[
                                                                     chosen_trees.starting_tree_type == "rand"].index)
        MSA_results.append(curr_tree_selection_metrics)
    return pd.concat(MSA_results)