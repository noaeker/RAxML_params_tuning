from side_code.config import *
from ML_pipeline.ML_pipeline_procedures import *
from ML_pipeline.ML_algorithms_and_hueristics import ML_model, print_model_statistics, train_test_validation_splits, \
    variable_importance
import pandas as pd
import os
import argparse
import numpy as np
import pickle
from sklearn.preprocessing import PolynomialFeatures


def get_MSA_clustering_and_threshold_results(curr_run_directory, msa_data, clusters_max_dist_options,
                                             max_starting_trees, total_MSA_level_features):
    curr_msa_data_per_tree = msa_data[
        ["msa_path", "starting_tree_ind", "starting_tree_object", "spr_radius", "spr_cutoff", "starting_tree_type",
         "starting_tree_bool"
            , "predicted_calibrated_failure_probabilities", "predicted_uncalibrated_failure_probabilities",
         "delta_ll_from_overall_msa_best_topology", "tree_clusters_ind",
         "is_global_max", "predicted_time", "normalized_relative_time"] + total_MSA_level_features]

    curr_msa_data_per_tree["log_failure_calibrated"] = np.log(
        curr_msa_data_per_tree["predicted_calibrated_failure_probabilities"])

    curr_msa_data_per_tree["predicted_uncalibrated_success_probability"] = curr_msa_data_per_tree[
        "predicted_calibrated_failure_probabilities"].apply(lambda x: 1 - x)
    curr_msa_data_per_tree["predicted_calibrated_success_probability"] = curr_msa_data_per_tree[
        "predicted_uncalibrated_failure_probabilities"].apply(lambda x: 1 - x)
    curr_msa_data_per_tree["failure_score"] = (curr_msa_data_per_tree[
                                                    "log_failure_calibrated"] /
                                                curr_msa_data_per_tree[
                                                    "predicted_time"]) * -1
    best_configuration_per_starting_tree = get_best_configuration_per_starting_tree(curr_msa_data_per_tree)
    clustering_const = 2 * (np.max(curr_msa_data_per_tree["feature_msa_n_seq"])) - 3

    best_parsimony_configuration_per_cluster_and_size = get_best_parsimony_config_per_cluster(curr_run_directory,
                                                                                              best_configuration_per_starting_tree.loc[
                                                                                                  best_configuration_per_starting_tree.starting_tree_type == "pars"],
                                                                                              normalizing_const=clustering_const,

                                                                                              max_dist_options=clusters_max_dist_options)
    all_configurations = []
    for max_dist in best_parsimony_configuration_per_cluster_and_size:
        possible_configurations = pd.concat(
            [best_parsimony_configuration_per_cluster_and_size[max_dist], best_configuration_per_starting_tree.loc[
                best_configuration_per_starting_tree.starting_tree_type == "rand"]])
        possible_configurations = possible_configurations.sort_values(by="predicted_uncalibrated_success_probability", ascending=False)
        possible_configurations["clusters_max_dist"] = max_dist
        possible_configurations["total_time_predicted"] = possible_configurations["predicted_time"].cumsum()
        possible_configurations["total_actual_time"] = possible_configurations["normalized_relative_time"].cumsum()
        possible_configurations["sum_of_predicted_success_probability_uncalibrated"] = possible_configurations[
            "predicted_uncalibrated_success_probability"].cumsum()
        possible_configurations["sum_of_predicted_success_probability_calibrated"] = possible_configurations[
            "predicted_calibrated_success_probability"].cumsum()
        possible_configurations["sum_of_log_failure_probability_calibrated"] = possible_configurations[
            "log_failure_calibrated"].cumsum()
        possible_configurations["status"] = possible_configurations["is_global_max"].cummax()
        possible_configurations["diff"] = possible_configurations["delta_ll_from_overall_msa_best_topology"].cummin()
        possible_configurations["n_trees_used"] = list(range(1, len(possible_configurations.index) + 1))
        possible_configurations["n_parsimony_trees_used"] = possible_configurations["starting_tree_bool"].cumsum()
        possible_configurations["n_random_trees_used"] = possible_configurations["n_trees_used"] - \
                                                         possible_configurations["n_parsimony_trees_used"]
        all_configurations.append(possible_configurations)
    return pd.concat(all_configurations)


def try_different_tree_selection_metodologies(curr_run_directory, data, clusters_max_dist_options, max_starting_trees,
                                              total_MSA_level_features):
    logging.info("Trying different procedures to select most promising trees")
    all_msa_validation_performance = []
    msa_paths = data["msa_path"].unique()
    for msa_path in msa_paths:
        msa_data = data.loc[data.msa_path == msa_path]
        curr_msa_results = get_MSA_clustering_and_threshold_results(curr_run_directory, msa_data,
                                                                    clusters_max_dist_options, max_starting_trees,
                                                                    total_MSA_level_features)
        all_msa_validation_performance.append(curr_msa_results)
    return pd.concat(all_msa_validation_performance)


def generate_multi_tree_data(full_data, X, time_model, error_model, args, total_MSA_level_features, out_path):
    if os.path.exists(out_path):
        data_dict = pickle.load(open(out_path, "rb"))
        logging.info("Using existing data dict")
        return data_dict
    time_data = time_model['selector'].transform(X)
    full_data["predicted_time"] = time_model['best_model'].predict(time_data)
    error_data = error_model['selector'].transform(X)
    full_data["predicted_calibrated_failure_probabilities"] = error_model['calibrated_model'].predict_proba(
        error_data)[:,
                                                              0]

    full_data["predicted_uncalibrated_failure_probabilities"] = error_model[
                                                                    'best_model'].predict_proba(
        error_data)[:,
                                                                0]
    multi_tree_data_df = try_different_tree_selection_metodologies(args.baseline_folder,
                                                                   full_data,
                                                                   max_starting_trees=args.max_starting_trees,
                                                                   clusters_max_dist_options=args.clusters_max_dist_options,
                                                                   total_MSA_level_features=total_MSA_level_features
                                                                   )
    multi_tree_data_df["predicted_iid_single_tree_failure_probability"] = np.exp(
        multi_tree_data_df["sum_of_log_failure_probability_calibrated"])
    multi_tree_data_df["predicted_iid_success_probabilities"] =multi_tree_data_df["predicted_iid_single_tree_failure_probability"].apply(lambda x: 1-x)
    final_ML_features = total_MSA_level_features + [
        "clusters_max_dist", #"sum_of_predicted_success_probability_uncalibrated",
        "sum_of_predicted_success_probability_calibrated",
        "predicted_iid_success_probabilities", "total_time_predicted", "n_random_trees_used",
        "n_parsimony_trees_used"]
        #["predicted_iid_success_probabilities","feature_msa_pypythia_msa_difficulty","clusters_max_dist","n_random_trees_used","n_parsimony_trees_used"]

    multitree_X = multi_tree_data_df[final_ML_features]
    #poly = PolynomialFeatures(2)
    #multitree_X = poly.fit_transform(multitree_X)
    groups = multi_tree_data_df["msa_path"]
    multitree_Y = multi_tree_data_df["status"]
    data_dict = {'full_multitree_data': multi_tree_data_df, 'multitree_X': multitree_X, 'multitree_Y': multitree_Y,
                 'groups': groups}
    pickle.dump(data_dict, open(out_path, "wb"))
    return data_dict


def get_multitree_performance_on_test_set_per_threshold(data_dict, args, time_model, error_model, total_MSA_level_features,
                                                        file_paths, thredhols = [0.8,0.85,0.9,0.95,0.98,0.99, 0.999, 0.9999,0.999999999999]):
    val_multi_tree_dict = generate_multi_tree_data(data_dict["full_validation_data"],
                                                   data_dict["X_val"], time_model, error_model,
                                                   args, total_MSA_level_features,
                                                   out_path=file_paths["validation_multi_tree_data"])
    total_error_model = ML_model(X_train=val_multi_tree_dict["multitree_X"],
                                 groups=val_multi_tree_dict["groups"],
                                 y_train=val_multi_tree_dict["multitree_Y"], n_jobs=1,
                                 path=file_paths["required_accuracy_model_path"],classifier= True,  calibrate= True
                                 )
    test_multi_tree_dict = generate_multi_tree_data(data_dict["full_test_data"],
                                                    data_dict["X_test"], time_model,
                                                    error_model,
                                                    args, total_MSA_level_features,
                                                    out_path=file_paths[
                                                        "test_multi_tree_data"])
    test_full_data = test_multi_tree_dict["full_multitree_data"]
    test_full_data['predicted_total_accuracy_calibrated'] = total_error_model[
                                                                'calibrated_model'].predict_proba(
        test_multi_tree_dict["multitree_X"])[:, 1]
    test_full_data['predicted_total_accuracy'] = total_error_model[
                                                     'best_model'].predict_proba(
        test_multi_tree_dict["multitree_X"])[:, 1]
    print_model_statistics(model=total_error_model,
                           test_X=test_multi_tree_dict["multitree_X"],
                           y_test=test_multi_tree_dict["multitree_Y"],
                           vi_path=file_paths["final_error_vi"],
                           is_classification=True, name="Final Error classification model", feature_importance = True)
    test_full_data["iid_expected_success_prob"] = test_full_data["predicted_iid_single_tree_failure_probability"].apply(lambda x: 1-x)
    accuracy_metrics = ['predicted_total_accuracy_calibrated','predicted_total_accuracy','iid_expected_success_prob']
    all_test_results = []
    for metric in accuracy_metrics:
        for threhold in thredhols:
            current_test_data_results = test_full_data.copy()
            current_test_data_results['accuracy_metric']= metric
            current_test_data_results['threshold'] = threhold
            current_test_data_results["max_accuracy"] = current_test_data_results.groupby("msa_path")[
                metric].transform(np.max)
            current_test_data_results = current_test_data_results[
                (current_test_data_results[metric] >= threhold) ]
            current_performance_on_test_set = current_test_data_results.sort_values("total_time_predicted").groupby('msa_path').head(1)
            current_performance_on_test_set["MSAs_included"] = len(current_performance_on_test_set.index)
            all_test_results.append(current_performance_on_test_set)
    return pd.concat(all_test_results)

