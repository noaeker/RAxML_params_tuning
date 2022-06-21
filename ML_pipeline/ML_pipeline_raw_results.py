import sys

if sys.platform == "linux" or sys.platform == "linux2":
    PROJECT_ROOT_DIRECRTORY = "/groups/pupko/noaeker/RAxML_params_tuning"
else:
    PROJECT_ROOT_DIRECRTORY = "/Users/noa/Workspace/RAxML_params_tuning"
sys.path.append(PROJECT_ROOT_DIRECRTORY)

from side_code.config import *
from side_code.file_handling import create_dir_if_not_exists
from side_code.MSA_manipulation import remove_env_path_prefix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import pickle
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, roc_auc_score
import numpy as np
import os
import argparse
import lightgbm
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV



# Mean absolute error (MAE)
def model_metrics(y_test, predictions, is_classification):
    if is_classification:
        return classification_report(y_test, predictions)
    return {"r2": r2_score(y_test, predictions), "MAE": mean_absolute_error(y_test, predictions),
            "MSE": mean_squared_error(y_test, predictions)
            }


def train_test_validation_splits(full_data, test_pct, val_pct):
    np.random.seed(SEED)
    msa_names = list(np.unique(full_data["msa_name"]))
    val_and_test_msas = np.random.choice(msa_names, size=int(len(msa_names) * (test_pct + val_pct)), replace=False)
    np.random.seed(SEED + 1)
    test_msas = np.random.choice(val_and_test_msas,
                                 size=int(len(val_and_test_msas) * (test_pct / (test_pct + val_pct))), replace=False)
    validation_msas = [name for name in val_and_test_msas if name not in test_msas]
    train_data = full_data[~full_data["msa_name"].isin(val_and_test_msas)]
    test_data = full_data[full_data["msa_name"].isin(test_msas)]
    validation_data = full_data[full_data["msa_name"].isin(validation_msas)]
    return train_data, test_data, validation_data


def variable_importance(X_train, rf_model):
    feats = {}  # a dict to hold feature_name: feature_importance
    for feature, importance in zip(X_train.columns, rf_model.feature_importances_):
        feats[feature] = importance  # add the name/value pair

    importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-importance'})
    importances.sort_values(by='Gini-importance', inplace=True)
    return importances


def error_classifier(X_train, y_train, path="gbm_error"):
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))

    param_grid = {
        'bootstrap': [True],
        'max_depth': [80, 90, 100, 110],
        'max_features': [2, 3],
        'min_samples_leaf': [3, 4, 5],
        'min_samples_split': [8, 10, 12],
        'n_estimators': [100, 200, 300, 1000]
    }
    rf = RandomForestClassifier()
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                               cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train.ravel())
    best_classifier = grid_search.best_estimator_

    # gbm_classifier = lightgbm.LGBMClassifier()
    # GS_res_gbm = GridSearchCV(gbm_classifier, lgbm_grid_param_space).fit(X_train, y_train.ravel())
    # clf_gbm = GS_res_gbm.best_estimator_
    # gbm_classifier.fit(X_train, y_train)
    pickle.dump(best_classifier, open(path, "wb"))
    return best_classifier


def time_regressor(X_train, y_train, path="gbm_time"):
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    param_grid = {
        'bootstrap': [True],
        'max_depth': [80, 90, 100, 110],
        'max_features': [2, 3],
        'min_samples_leaf': [3, 4, 5],
        'min_samples_split': [8, 10, 12],
        'n_estimators': [100, 200, 300, 1000]
    }
    rf_time = RandomForestRegressor()
    grid_search = GridSearchCV(estimator=rf_time, param_grid=param_grid,
                               cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train.ravel())
    best_regressor = grid_search.best_estimator_
    # Calculate the absolute errors
    pickle.dump(best_regressor, open(path, "wb"))
    return best_regressor


def take_top_n_most_promising_trees(enriched_test_data):
    results_per_msa = []
    for msa_path in enriched_test_data["msa_path"].unique():
        curr_msa_data_per_tree = enriched_test_data[enriched_test_data["msa_path"] == msa_path][
            ["msa_path", "starting_tree_ind", "spr_radius", "spr_cutoff", "starting_tree_type",
             "predicted_failure_probabilities",
             "predicted_status", "is_global_max", "predicted_time", "relative_time"]]
        curr_msa_data_per_tree.sort_values(
            by=['starting_tree_ind', 'predicted_failure_probabilities', 'predicted_time'],
            inplace=True)
        best_configuration_per_starting_tree = curr_msa_data_per_tree.groupby('starting_tree_ind').head(1)
        required_success_probability = 0.99
        for n_trees in range(1, 41):
            current_tree_used = best_configuration_per_starting_tree.head(n_trees)
            current_failure_prob_list = list(current_tree_used['predicted_failure_probabilities'])
            current_success_prob_overall = 1 - np.prod(current_failure_prob_list)
            current_option = current_tree_used.groupby(["msa_path"]).agg(
                total_time_predicted=('predicted_time', np.sum), total_actual_time=('relative_time', np.sum),
                status=('is_global_max', np.max)).reset_index()
            if current_success_prob_overall >= required_success_probability:
                break
        current_option["n_trees_used"] = n_trees
        results_per_msa.append(current_option)
    final_performance_df = pd.concat(results_per_msa)
    final_performance_df.to_csv(f"{READY_RAW_DATA}/c_30_70/final_performance.tsv", sep=CSV_SEP)
    return final_performance_df


def train_models(data_dict, error_model_path, time_model_path):
    logging.info("About to calculate RF for modelling error")
    error_model = error_classifier(data_dict["X_train"], data_dict["y_train_err"],error_model_path)
    err_var_impt = variable_importance(data_dict["X_train"], error_model)
    logging.info(f"Error RF variable importance: \n {err_var_impt}")
    predicted_success = error_model.predict(data_dict["X_test"])
    predicted_failure_probabilities = error_model.predict_proba(data_dict["X_test"])[:, 0]
    err_test_metrics = model_metrics(data_dict["y_test_err"], predicted_success, is_classification=True)
    logging.info(f"Error RF metrics: \n {err_test_metrics}")
    time_model = time_regressor(data_dict["X_train"], data_dict["y_train_time"],time_model_path)
    time_var_impt = variable_importance(data_dict["X_train"], time_model)
    logging.info(f"Time RF variable importance: \n {time_var_impt}")
    predicted_time = time_model.predict(data_dict["X_test"])
    time_test_metrics = model_metrics(data_dict["y_test_time"], predicted_time, is_classification=False)
    logging.info(f"Time_test_metrics: \n {time_test_metrics}")

    enriched_test_data = data_dict["full_test_data"].copy()
    enriched_test_data["predicted_failure_probabilities"] = predicted_failure_probabilities
    enriched_test_data["predicted_status"] = predicted_success
    enriched_test_data["predicted_time"] = predicted_time
    final_performance_df = take_top_n_most_promising_trees(enriched_test_data)

    return error_model, time_model, final_performance_df


def split_to_train_and_test(full_data, data_feature_names, search_feature_names):
    train_data, test_data, validation_data = train_test_validation_splits(
        full_data, test_pct=0.3, val_pct=0)
    X_train = train_data[data_feature_names + search_feature_names]
    y_train_err = train_data["is_global_max"]
    y_train_time = train_data["relative_time"]
    X_test = test_data[data_feature_names + search_feature_names]
    y_test_err = test_data["is_global_max"]
    y_test_time = test_data["relative_time"]
    return {"X_train": X_train, "y_train_err": y_train_err, "y_train_time": y_train_time, "X_test": X_test,
            "y_train_surv": train_data[["is_global_max", "relative_time"]].to_records(index=False),
            "y_test_err": y_test_err, "y_test_time": y_test_time,
            "y_test_surv": test_data[["is_global_max", "relative_time"]].to_records(index=False),
            "full_test_data": test_data}


def edit_data(data, epsilon):
    data["is_global_max"] = (data["delta_ll_from_overall_msa_best_topology"] <= epsilon).astype('int')
    # data["is_global_max"] = data.groupby('msa_path').transform(lambda x: x<= x.quantile(0.1))["delta_ll_from_overall_msa_best_topology"]
    data["relative_time"] = data["elapsed_running_time"] / data["test_norm_const"]
    data["msa_name"] = data["msa_path"].apply(lambda s: remove_env_path_prefix(s))
    data["starting_tree_bool"] = data["starting_tree_type"] == "pars"
    data["feature_diff_vs_best_tree"] = \
        data.groupby(['msa_path', 'starting_tree_type']).transform(lambda x: (x - x.max()))["feature_optimized_ll"]
    data["feature_brlen_opt_effect"] = data["feature_optimized_ll"] - data["starting_tree_ll"]
    # data["feature_msa_mean_diff_vs_best_tree"] = data.groupby(['msa_path']).transform(lambda x: x.mean())["feature_diff_vs_best_tree"]
    # data["feature_starting_tree_ll_normalized"] = \
    #     data.groupby(['msa_path', 'starting_tree_type']).transform(lambda x: (x - x.mean()) / x.std())[
    #         "starting_tree_ll"]
    # data["feature_starting_tree_ll_optimized_normalized"] = \
    #     data.groupby(['msa_path', 'starting_tree_type']).transform(lambda x: (x - x.mean()) / x.std())[
    #         "feature_optimized_ll"]
    data["feature_mean_rf_distance_scaled"] = data["feature_mean_rf_distance"] / data["feature_n_seq"]
    # data["feature_tree_MAD_normalized"] = data.groupby('msa_path').transform(lambda x: (x - x.mean()) / x.std())["feature_tree_MAD"]
    # data["feature_largest_branch_length_normalized"] = data.groupby('msa_path').transform(lambda x: (x - x.mean()) / x.std())[
    #     "feature_largest_branch_length"]
    data["normalized_time"] = data.groupby('msa_path').transform(lambda x: (x - x.mean()) / x.std())["relative_time"]


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
            by=["msa_path", "best_msa_ll"]).agg(
            {"delta_ll_from_overall_msa_best_topology": ['min'], "is_global_max": ['max'], 'relative_time': ['sum']})
        run_metrics.columns = ["curr_sample_Err", "curr_sample_is_global_max", "curr_sample_total_time"]
        run_metrics.reset_index(inplace=True)
        if default_results.empty:
            default_results = run_metrics.copy()
        else:
            default_results = pd.concat([default_results, run_metrics])
    return default_results


def main():
    epsilon = 0.1
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline_folder',action='store', type=str,default=f"{READY_RAW_DATA}/c_30_70")
    parser.add_argument('--n_sample_points', action='store', type=int,
                        default=500)
    args = parser.parse_args()
    features_path = f"{args.baseline_folder}/features{CSV_SUFFIX}"
    ML_edited_features_path = f"{args.baseline_folder}/ML_edited_features{CSV_SUFFIX}"
    default_path = f"{args.baseline_folder}/default_sampling{CSV_SUFFIX}"
    final_performance_path =f"{args.baseline_folder}/final_performance{CSV_SUFFIX}"
    error_model_path = f"{args.baseline_folder}/error.model"
    time_model_path = f"{args.baseline_folder}/time.model"
    final_comparison_path = f"{args.baseline_folder}/final_performance_comp{CSV_SUFFIX}"
    log_file = f"{args.baseline_folder}/ML_log_file.log"


    data = pd.read_csv(features_path, sep=CSV_SEP)
    edit_data(data, epsilon)
    data.to_csv(ML_edited_features_path, sep=CSV_SEP)
    # full_data = full_data.replace([np.inf, -np.inf,np.nan], -1)
    logging_level = logging.INFO
    logging.basicConfig(filename=log_file, level=logging_level)
    if os.path.exists(final_performance_path):
        logging.info(f"Using our existing performance in {final_performance_path}")
        final_performance_df = pd.read_csv(final_performance_path, sep=CSV_SEP)
    else:
        logging.info("Estimating time and error models from beggining")
        msa_features = [col for col in data.columns if
                        col.startswith("feature_") and col not in ["feature_msa_path", "feature_msa_name",
                                                                   "feature_msa_type"]]
        search_features = ['spr_radius', 'spr_cutoff', 'starting_tree_bool', "starting_tree_ll"]
        data_dict = split_to_train_and_test(data, msa_features, search_features)
        rf_mod_err, rf_mod_time, final_performance_df = train_models(data_dict,
                                                                     error_model_path, time_model_path)
    if not os.path.exists(default_path):
        logging.info(f"Using existing default data in {default_path}")
        default_data = data[data["type"] == "default"]
        default_data_performance = get_average_results_on_default_configurations_per_msa(default_data,
                                                                                         n_sample_points=args.n_sample_points,
                                                                                         seed=SEED
                                                                                         )
    else:
        logging.info("Generating default data from beggining")
        default_data_performance = pd.read_csv(default_path, sep=CSV_SEP)
    aggregated_default_results = default_data_performance.groupby(by=["msa_path"]).agg(
        default_mean_is_global_max=('curr_sample_is_global_max', np.mean),
        default_mean_time=("curr_sample_total_time", np.mean)).reset_index()

    comp = final_performance_df.merge(aggregated_default_results, on="msa_path")
    comp.to_csv(final_comparison_path, sep=CSV_SEP)


if __name__ == "__main__":
    main()
