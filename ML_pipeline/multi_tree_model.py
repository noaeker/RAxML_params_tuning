from side_code.config import *
from ML_pipeline.ML_pipeline_procedures import *
from ML_pipeline.ML_algorithms_and_hueristics import *
from ML_pipeline.ML_algorithms_and_hueristics import ML_model, print_model_statistics, train_test_validation_splits, \
    variable_importance
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import PolynomialFeatures
from scipy.optimize import curve_fit
import seaborn as sns


def extend_multi_tree_metrics(possible_configurations):
    possible_configurations["sum_of_log_failure_probability_calibrated"] = possible_configurations[
        "log_failure_calibrated"].cumsum()
    possible_configurations["sum_of_log_failure_probability_calibrated_pars"] = possible_configurations[
        "log_failure_calibrated_pars"].cumsum()
    possible_configurations["sum_of_log_failure_probability_calibrated_rand"] = possible_configurations[
        "log_failure_calibrated_rand"].cumsum()
    possible_configurations["status"] = possible_configurations["is_global_max"].cummax()
    possible_configurations["total_time_predicted"] = possible_configurations["predicted_time"].cumsum()
    possible_configurations["total_actual_time"] = possible_configurations["normalized_relative_time"].cumsum()
    possible_configurations["diff"] = possible_configurations["delta_ll_from_overall_msa_best_topology"].cummin()
    possible_configurations["n_trees_used"] = list(range(1, len(possible_configurations.index) + 1))
    possible_configurations["n_parsimony_trees_used"] = possible_configurations["starting_tree_bool"].cumsum()
    possible_configurations["n_random_trees_used"] = possible_configurations["n_trees_used"] - \
                                                     possible_configurations["n_parsimony_trees_used"]
    possible_configurations["predicted_iid_single_tree_failure_probability"] = np.exp(
        possible_configurations["sum_of_log_failure_probability_calibrated"])
    possible_configurations["predicted_iid_single_tree_failure_probability_pars"] = np.exp(
        possible_configurations["sum_of_log_failure_probability_calibrated_pars"])
    possible_configurations["predicted_iid_single_tree_failure_probability_rand"] = np.exp(
        possible_configurations["sum_of_log_failure_probability_calibrated_rand"])
    possible_configurations["predicted_iid_success_probabilities"] = possible_configurations[
        "predicted_iid_single_tree_failure_probability"].apply(lambda x: 1 - x)
    possible_configurations["predicted_iid_success_probabilities_pars"] = possible_configurations[
        "predicted_iid_single_tree_failure_probability_pars"].apply(lambda x: 1 - x)
    possible_configurations["predicted_iid_success_probabilities_rand"] = possible_configurations[
        "predicted_iid_single_tree_failure_probability_rand"].apply(lambda x: 1 - x)
    return possible_configurations



def get_MSA_multi_tree_metrics(data, total_MSA_level_features, shuffle = True):
    logging.info("Trying different procedures to select most promising trees")
    all_msa_validation_performance = []
    msa_paths = data["msa_path"].unique()
    for msa_path in msa_paths:
        msa_data = data.loc[data.msa_path == msa_path]
        curr_msa_data_per_tree = msa_data[
            ["msa_path", "starting_tree_ind", "starting_tree_object", "spr_radius", "spr_cutoff", "starting_tree_type",
             "starting_tree_bool", "spr_radius","spr_cutoff","equal_to_default_config"

                , "predicted_calibrated_failure_probabilities", "predicted_uncalibrated_failure_probabilities",
             "delta_ll_from_overall_msa_best_topology", "tree_clusters_ind",
             "is_global_max", "predicted_time", "normalized_relative_time"] + total_MSA_level_features]

        curr_msa_data_per_tree["log_failure_calibrated"] = np.log(
            curr_msa_data_per_tree["predicted_calibrated_failure_probabilities"])


        curr_msa_data_per_tree["log_failure_calibrated_pars"] = np.where(curr_msa_data_per_tree["starting_tree_type"] == 'pars', curr_msa_data_per_tree["log_failure_calibrated"], 0)
        curr_msa_data_per_tree["log_failure_calibrated_rand"] = np.where(
            curr_msa_data_per_tree["starting_tree_type"] == 'rand', curr_msa_data_per_tree["log_failure_calibrated"], 0)
        curr_msa_data_per_tree["predicted_uncalibrated_success_probability"] = curr_msa_data_per_tree[
            "predicted_calibrated_failure_probabilities"].apply(lambda x: 1 - x)
        curr_msa_data_per_tree["predicted_calibrated_success_probability"] = curr_msa_data_per_tree[
            "predicted_uncalibrated_failure_probabilities"].apply(lambda x: 1 - x)
        # curr_msa_data_per_tree["failure_score"] = (curr_msa_data_per_tree[
        #                                                 "log_failure_calibrated"] /
        #                                           f  curr_msa_data_per_tree[
        #                                                 "predicted_time"]) * -1
        best_configuration_per_starting_tree = get_best_configuration_per_starting_tree(curr_msa_data_per_tree)
        if not shuffle:
            possible_configurations = best_configuration_per_starting_tree.sort_values(
               by="predicted_uncalibrated_success_probability",
               ascending=False)
            #possible_configurations = best_configuration_per_starting_tree.sample(frac=1)
            all_msa_validation_performance.append(extend_multi_tree_metrics(possible_configurations))
        else:
            for i in range(30):
                possible_configurations = best_configuration_per_starting_tree.sample(frac=1)
                all_msa_validation_performance.append(extend_multi_tree_metrics(possible_configurations))
    return pd.concat(all_msa_validation_performance)


def multi_tree_data_pipeline(full_data, X, time_model, error_model, total_MSA_level_features,singletree_out_path, multitree_out_path, shuffle):
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
    full_data.to_csv(singletree_out_path, sep =CSV_SEP)
    multi_tree_data_df = get_MSA_multi_tree_metrics(
        full_data,
        total_MSA_level_features=total_MSA_level_features, shuffle = shuffle
    )

    multi_tree_data_df.to_csv(multitree_out_path, sep = CSV_SEP)
    return multi_tree_data_df


def logistic_regression_pipeline(val_multi_tree_data,test_multi_tree_data):
    probability_col = ["sum_of_log_failure_probability_calibrated","sum_of_log_failure_probability_calibrated_pars","sum_of_log_failure_probability_calibrated_rand","feature_msa_pypythia_msa_difficulty"]
    poly = PolynomialFeatures(2)
    X_validation = val_multi_tree_data[probability_col]
    X_validation = poly.fit_transform(X_validation)
    y_validation = val_multi_tree_data["status"]
    group_splitter = list(
        GroupKFold(n_splits=5).split(X_validation, y_validation.ravel(), groups=val_multi_tree_data["msa_path"]))
    logistic_model = LogisticRegressionCV(cv=group_splitter,max_iter=1000, scoring = 'roc_auc').fit(y=y_validation, X=X_validation)

    X_test = test_multi_tree_data[probability_col]
    X_test = poly.fit_transform(X_test)
    y_test = test_multi_tree_data["status"]
    test_multi_tree_data['predicted_total_accuracy_calibrated_logistic'] = logistic_model.predict_proba(X_test
                                                                                               )[:, 1]
    # predicted = test_multi_tree_data['predicted_total_accuracy_calibrated'].apply(lambda x: int(x>0.5))
    predicted = logistic_model.predict(X_test)
    test_metrics = model_metrics(y_test, predicted, test_multi_tree_data['predicted_total_accuracy_calibrated_logistic'],
                                 is_classification=True)
    logging.info(f"Multitree logistic model metrics: {test_metrics}")


def isotonic_regression_pipeline(val_multi_tree_data,test_multi_tree_data):

    probability_col = "predicted_iid_success_probabilities"
    val_multi_tree_data['binned_pypythia_difficulty'] = pd.cut(val_multi_tree_data['feature_msa_pypythia_msa_difficulty'],
                                                              bins=[0,0.25,0.5,1]
                                                              )
    val_multi_tree_data["binned_iid_probabilities"] = pd.cut(val_multi_tree_data[probability_col],
                                                             bins=50
                                                             )

    val_multi_tree_data["predicted_probabilities"] = val_multi_tree_data.groupby("binned_iid_probabilities")[
        'predicted_iid_success_probabilities'].transform(np.mean)
    val_multi_tree_data["actual_probabilities"] = val_multi_tree_data.groupby("binned_iid_probabilities")[
        'status'].transform(np.mean)
    ir = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
    X_validation = val_multi_tree_data["predicted_probabilities"]
    y_validation = val_multi_tree_data["actual_probabilities"]
    # sns.lineplot(data=val_multi_tree_data, x="predicted_probabilities", y="actual_probabilities", hue="binned_pypythia_difficulty")
    # plt.show()
    # for b in val_multi_tree_data['binned_pypythia_difficulty'].unique():
    #     data = val_multi_tree_data[val_multi_tree_data['binned_pypythia_difficulty']==b]
    #     sns.scatterplot(data=data, x="predicted_probabilities", y="actual_probabilities",
    #                  )
    #     plt.show()
    ir_model = ir.fit(X_validation, y_validation)
    X_test = list(test_multi_tree_data[probability_col])
    test_multi_tree_data['predicted_total_accuracy_calibrated_isotonic'] = ir_model.predict(X_test)
    predicted = test_multi_tree_data['predicted_total_accuracy_calibrated_isotonic'].apply(lambda x: int(x>0.5))
    test_metrics = model_metrics(test_multi_tree_data["status"] ,predicted, test_multi_tree_data['predicted_total_accuracy_calibrated_isotonic'],
                                 is_classification=True)
    logging.info(f"Multitree isotonic model metrics: {test_metrics}")



def get_multitree_performance_on_test_set_per_threshold(data_dict, args, time_model, error_model,
                                                        total_MSA_level_features,
                                                        file_paths
                                                        ):
    if os.path.exists(file_paths["validation_multi_tree_data"]):
        val_multi_tree_data = pd.read_csv(file_paths["validation_multi_tree_data"], sep = CSV_SEP)
    else:
        val_multi_tree_data = multi_tree_data_pipeline(data_dict["full_validation_data"],
                                                       data_dict["X_val"], time_model, error_model,
                                                       total_MSA_level_features,
                                                       singletree_out_path = file_paths["validation_single_tree_data"],
                                                       multitree_out_path=file_paths["validation_multi_tree_data"], shuffle = True)

    if os.path.exists(file_paths["test_multi_tree_data"]):
        test_multi_tree_data = pd.read_csv(file_paths["test_multi_tree_data"], sep=CSV_SEP)
    else:
        test_multi_tree_data = multi_tree_data_pipeline(data_dict["full_test_data"],
                                                        data_dict["X_test"], time_model,
                                                        error_model,
                                                        total_MSA_level_features,
                                                        singletree_out_path=file_paths["test_single_tree_data"],
                                                        multitree_out_path=file_paths[
                                                            "test_multi_tree_data"], shuffle = False)
    logistic_regression_pipeline(val_multi_tree_data, test_multi_tree_data)
    isotonic_regression_pipeline(val_multi_tree_data,test_multi_tree_data)
    all_test_results = []
    logging.info(f"Number of MSAs in test set = {len(test_multi_tree_data['msa_path'].unique())}")
    best_result_per_MSA = test_multi_tree_data[
        test_multi_tree_data["status"] == 1].sort_values(
        "total_time_predicted").groupby("msa_path").head(1)[["msa_path","total_actual_time", "predicted_total_accuracy_calibrated_isotonic"]].rename(
        columns={'total_actual_time': 'optimal_time', 'predicted_total_accuracy_calibrated_isotonic':'optimal_isotonic_threshold'})
    logging.info(f"Number of MSAs in test set with potenetial global maxima = {len(best_result_per_MSA['msa_path'].unique())}")
    for threhold in [0,0.6,0.7,0.8,0.9,0.95,0.99]:
        for metric in ["predicted_total_accuracy_calibrated_isotonic","predicted_total_accuracy_calibrated_logistic"]:
            test_multi_tree_data["max_accuracy"] = test_multi_tree_data.groupby("msa_path")[metric].transform(np.max)
            results_per_MSA = test_multi_tree_data[
                (test_multi_tree_data[metric] >= threhold)].sort_values(
                "total_time_predicted").groupby("msa_path").head(1)
            results_per_MSA = results_per_MSA.merge(best_result_per_MSA, on = "msa_path", how = 'left')
            results_per_MSA["threshold"] = threhold
            results_per_MSA["MSAs_included"] = len(results_per_MSA.index)
            results_per_MSA["metric"] = metric
            all_test_results.append(results_per_MSA)
    return pd.concat([df for df in all_test_results if len(df.index)>0])
#(test_multi_tree_data[metric]>=test_multi_tree_data["max_accuracy"])
