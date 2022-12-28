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
from sklearn import svm
from sklearn.kernel_approximation import Nystroem
from sklearn.svm import SVC
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.linear_model import CoxPHSurvivalAnalysis


def extend_multi_tree_metrics(possible_configurations):
    possible_configurations["status"] = possible_configurations["is_global_max"].cummax()
    possible_configurations["total_time_predicted"] = possible_configurations["predicted_time"].cumsum()
    possible_configurations["total_actual_time"] = possible_configurations["normalized_relative_time"].cumsum()
    possible_configurations["diff"] = possible_configurations["delta_ll_from_overall_msa_best_topology"].cummin()
    possible_configurations["n_trees_used"] = list(range(1, len(possible_configurations.index) + 1))
    possible_configurations["predicted_iid_single_tree_failure_probability"] = possible_configurations["predicted_calibrated_failure_probabilities"].cumprod()
    possible_configurations["predicted_iid_success_probabilities"] = possible_configurations[
        "predicted_iid_single_tree_failure_probability"].apply(lambda x: 1 - x)
    possible_configurations["sum_of_calibrated_success_probabilities"] = possible_configurations['predicted_calibrated_success_probability'].cumsum()
    possible_configurations["sum_of_uncalibrated_success_probabilities"] = possible_configurations['predicted_uncalibrated_success_probability'].cumsum()
    possible_configurations['min_mds_0'] = possible_configurations['feature_mds_False_pca_0_3_spr_enriched'].cummin()
    possible_configurations['max_mds_0'] = possible_configurations['feature_mds_False_pca_0_3_spr_enriched'].cummax()
    return possible_configurations



def get_MSA_multi_tree_metrics(data, total_MSA_level_features, shuffle = True):
    logging.info("Trying different procedures to select most promising trees")
    all_msa_validation_performance = []
    msa_paths = data["msa_path"].unique()
    for msa_path in msa_paths:
        for starting_tree_type in ('pars','rand'):
            msa_data = data.loc[(data.msa_path == msa_path)&(data.starting_tree_type==starting_tree_type)]
            relevant_msa_data = msa_data[
                ["msa_path", "starting_tree_ind", "starting_tree_object", "spr_radius", "spr_cutoff", "starting_tree_type",
                 "starting_tree_bool","feature_tree_optimized_ll", "spr_radius","spr_cutoff","equal_to_default_config",
                 "feature_mds_False_pca_0_3_spr_enriched"

                    , "predicted_calibrated_failure_probabilities", "predicted_uncalibrated_failure_probabilities",
                 "delta_ll_from_overall_msa_best_topology", "tree_clusters_ind",
                 "is_global_max", "predicted_time", "normalized_relative_time","final_tree_topology","final_ll"] + total_MSA_level_features]

            #relevant_msa_data["log_failure_calibrated"] = np.log(
            #    relevant_msa_data["predicted_calibrated_failure_probabilities"])
            #relevant_msa_data["predicted_calibrated_success_probabilities"] =

            relevant_msa_data["predicted_uncalibrated_success_probability"] = relevant_msa_data[
                "predicted_uncalibrated_failure_probabilities"].apply(lambda x: 1 - x)
            relevant_msa_data["predicted_calibrated_success_probability"] = relevant_msa_data[
                "predicted_calibrated_failure_probabilities"].apply(lambda x: 1 - x)
            best_configuration_per_starting_tree = get_best_configuration_per_starting_tree(relevant_msa_data)
            if not shuffle:
                possible_configurations = best_configuration_per_starting_tree.sort_values(
                   by="predicted_uncalibrated_failure_probabilities"
                   )
                #possible_configurations = best_configuration_per_starting_tree.sample(frac=1)
                all_msa_validation_performance.append(extend_multi_tree_metrics(possible_configurations))
            else:
                for i in range(100):
                    possible_configurations = best_configuration_per_starting_tree.sample(frac=1)
                    possible_configurations["simulation_ind"] = i
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
    #multi_tree_data_df = get_MSA_multi_tree_metrics(
    #    full_data,
    #    total_MSA_level_features=total_MSA_level_features, shuffle = shuffle
    #)

    #multi_tree_data_df.to_csv(multitree_out_path, sep = CSV_SEP)
    #return multi_tree_data_df


# def multi_tree_final_model_pipeline(val_multi_tree_data, test_multi_tree_data):
#     #"predicted_iid_success_probabilities_rand","predicted_iid_success_probabilities_pars"
#     probability_col = ["n_trees_used","sum_of_calibrated_success_probabilities","predicted_iid_success_probabilities_pars","predicted_iid_success_probabilities_rand","feature_msa_pypythia_msa_difficulty"] #mds_pc_0_variance
#     val_multi_tree_data["status_bool"] = val_multi_tree_data["status"].astype(bool)
#     for starting_tree_type in ("pars", "rand"):
#         mask_tree = val_multi_tree_data["starting_tree_type"] == starting_tree_type
#         time_tree, survival_prob_tree = kaplan_meier_estimator(
#             val_multi_tree_data["status_bool"][mask_tree],
#             val_multi_tree_data["n_trees_used"][mask_tree])
#
#         plt.step(time_tree, survival_prob_tree, where="post",
#                  label="Tree = %s" % starting_tree_type)
#
#     plt.ylabel("est. probability of survival $\hat{S}(t)$")
#     plt.xlabel("time $t$")
#     plt.legend(loc="best")
#
#     time, survival_prob = kaplan_meier_estimator(val_multi_tree_data["status_bool"], val_multi_tree_data["n_trees_used"])
#     plt.show()
#
#
#
#
#
#     poly = PolynomialFeatures(2)
#     X_validation = val_multi_tree_data[probability_col]
#     X_validation["best_diff_per_group"] = X_validation.groupby("msa_path")["diff"].transform(min)
#     X_validation["min_time_per_group"] = X_validation.groupby("msa_path")["diff"].transform(min)
#     X_validation = X_validation[(X_validation["diff"]>X_validation["best_diff_per_group"]) | (X_validation["diff"]>X_validation["best_diff_per_group"])]
#     #X_validation=poly.fit_transform(X_validation)
#     #X_validation = X_validation.sort_values('diff').groupby('msa_path').head()
#     #X_validation = poly.fit_transform(X_validation)
#     y_validation = val_multi_tree_data[""]
#     group_splitter = list(
#         GroupKFold(n_splits=5).split(X_validation, y_validation.ravel(), groups=val_multi_tree_data["msa_path"]))
#
#     model = make_pipeline(StandardScaler(), LogisticRegressionCV(cv=group_splitter,max_iter=1000, scoring = 'roc_auc'))#SGDClassifier(max_iter=1000, tol=1e-3, loss = 'modified_huber')
#     model.fit(X_validation,y_validation)
#     # model = ML_model(X_train=X_validation, groups=val_multi_tree_data["msa_path"], y_train=y_validation,
#     #          n_jobs=4,
#     #          path=None, classifier=False)
#     #for coef,feature in zip(model._final_estimator.coef_, probability_col):
#     #   logging.info(f"Feature {feature} has coefficient {coef}")
#     X_test = test_multi_tree_data[probability_col]
#     #X_test = poly.fit_transform(X_test)
#     #X_test = poly.fit_transform(X_test)
#     y_test = test_multi_tree_data[y]
#     test_multi_tree_data['predicted_total_accuracy_calibrated_logistic'] = model.predict_proba(X_test)[:, 1]
#     #
#     # predicted = test_multi_tree_data['predicted_total_accuracy_calibrated'].apply(lambda x: int(x>0.5))
#     predicted = model.predict(X_test)
#     test_metrics = model_metrics(y_test, predicted, test_multi_tree_data['predicted_total_accuracy_calibrated_logistic'],
#                                 is_classification=True, groups = test_multi_tree_data["msa_path"])
#     #logging.info(f"Multitree logistic model metrics: {test_metrics}")
#     #print_model_statistics(model, X_test, test_multi_tree_data["msa_path"], y_test, is_classification = False, vi_path = None, name = None,
#     #                       feature_importance=True)
#

# def test_required_threshold(val_multi_tree_data,test_multi_tree_data):
#     #probability_col = ["predicted_iid_success_probabilities",
#     #                   "feature_msa_pypythia_msa_difficulty","i" ]  # mds_pc_0_variance
#     #X_validation = val_multi_tree_data[probability_col]
#     X_validation = val_multi_tree_data.loc[val_multi_tree_data.status==1].groupby(['msa_path',"feature_msa_pypythia_msa_difficulty","i"])["predicted_iid_success_probabilities"].transform(min)
#     X_validation
#     # X_validation = poly.fit_transform(X_validation)
#     y_validation = val_multi_tree_data["status"]
#     group_splitter = list(
#         GroupKFold(n_splits=5).split(X_validation, y_validation.ravel(), groups=val_multi_tree_data["msa_path"]))
#
#     model = lightgbm.LGBMRegressor()  # SGDClassifier(max_iter=1000, tol=1e-3, loss = 'modified_huber')
#
#     # calibrated_model = CalibratedClassifierCV(base_estimator=model, cv=group_splitter, method='isotonic')
#     model.fit(y=y_validation, X=X_validation)
#     for coef, feature in zip(model._final_estimator.coef_, probability_col):
#         logging.info(f"Feature {feature} has coefficient {coef}")
#     test_multi_tree_data["log_n_trees"] = np.log(test_multi_tree_data["n_trees_used"])
#     X_test = test_multi_tree_data[probability_col]
#     # X_test = poly.fit_transform(X_test)
#     y_test = test_multi_tree_data["status"]
#     test_multi_tree_data['predicted_total_accuracy_calibrated_logistic'] = model.predict_proba(X_test
#                                                                                                )[:, 1]
#     # predicted = test_multi_tree_data['predicted_total_accuracy_calibrated'].apply(lambda x: int(x>0.5))
#     predicted = model.predict(X_test)
#     test_metrics = model_metrics(y_test, predicted,
#                                  test_multi_tree_data['predicted_total_accuracy_calibrated_logistic'],
#                                  is_classification=True, groups=test_multi_tree_data["msa_path"])
#     logging.info(f"Multitree logistic model metrics: {test_metrics}")


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
                                 is_classification=True,groups=test_multi_tree_data["msa_path"])
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
                                                       multitree_out_path=file_paths["validation_multi_tree_data"], shuffle =False)

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
    multi_tree_final_model_pipeline(val_multi_tree_data, test_multi_tree_data)
    isotonic_regression_pipeline(val_multi_tree_data,test_multi_tree_data)
    #test_required_threshold(val_multi_tree_data,test_multi_tree_data)
    test_multi_tree_data.to_csv(file_paths["test_multi_tree_data_with_predictions"])
    all_test_results = []
    logging.info(f"Number of MSAs in test set = {len(test_multi_tree_data['msa_path'].unique())}")
    best_result_per_MSA = test_multi_tree_data[
        test_multi_tree_data["status"] == 1].sort_values(
        "total_time_predicted").groupby("msa_path").head(1)[["msa_path","total_actual_time", "predicted_total_accuracy_calibrated_isotonic"]].rename(
        columns={'total_actual_time': 'optimal_time', 'predicted_total_accuracy_calibrated_isotonic':'optimal_isotonic_threshold'})
    logging.info(f"Number of MSAs in test set with potenetial global maxima = {len(best_result_per_MSA['msa_path'].unique())}")
    metric = "predicted_total_accuracy_calibrated_logistic"
    test_multi_tree_data["max_accuracy"] = test_multi_tree_data.groupby("msa_path")[metric].transform(np.max)
    for threhold in [0.8,0.85,0.9,0.95,0.99]:
        results_per_MSA = test_multi_tree_data[
            (test_multi_tree_data[metric] >= threhold)].sort_values(
            "total_time_predicted").groupby("msa_path").head(1)
        results_per_MSA["threshold"] = threhold
        results_per_MSA["MSAs_included"] = len(results_per_MSA.index)
        results_per_MSA["metric"] = metric
        all_test_results.append(results_per_MSA)
    best_balance_per_MSA = test_multi_tree_data[test_multi_tree_data[metric]>0.95*test_multi_tree_data["max_accuracy"]].sort_values(
            "total_time_predicted").groupby('msa_path').head(1)
    best_balance_per_MSA["threshold"] = -1
    best_balance_per_MSA["MSAs_included"] = len(best_balance_per_MSA.index)
    best_balance_per_MSA["metric"] = metric
    all_test_results.append(best_balance_per_MSA)
    all = pd.concat([df for df in all_test_results if len(df.index)>0])
    return all.merge(best_result_per_MSA, on = "msa_path", how = 'left')
#(test_multi_tree_data[metric]>=test_multi_tree_data["max_accuracy"])
