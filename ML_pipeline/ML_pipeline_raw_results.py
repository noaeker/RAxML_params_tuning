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


def classifier(X_train, y_train, path):
    gbm_cl = lightgbm.LGBMClassifier()
    gbm_cl.fit(X_train, y_train)
    return gbm_cl


def regressor(X_train, y_train, path):
    gbm_reg = lightgbm.LGBMRegressor()
    gbm_reg.fit(X_train, y_train)
    # Calculate the absolute errors
    return gbm_reg


def plot_full_data_metrics(full_data, epsilon):
    first_msa = list(full_data["msa_name"].unique())[2]
    full_data = full_data[full_data["msa_name"] == first_msa]

    allowed_actual_error_data = full_data[(full_data["pct_global"] > epsilon)]
    allowed_actual_error_data["total_starting_points"] = allowed_actual_error_data["n_parsimony"] + \
                                                         allowed_actual_error_data["n_random"]
    actual_best_time_per_msa = pd.merge(allowed_actual_error_data,
                                        allowed_actual_error_data.groupby("msa_name").agg(
                                            best_actual_running_time=(
                                                'running_time_vs_default', max)).reset_index(),
                                        left_on=["msa_name", "running_time_vs_default"],
                                        right_on=["msa_name", "best_actual_running_time"])
    actual_best_time_per_msa = actual_best_time_per_msa[actual_best_time_per_msa["best_actual_running_time"] > 1]
    # sns.set(font_scale=1.5)
    # plt.show()
    # sns.set(font_scale=1.5)
    # ax = sns.histplot(x = actual_best_time_per_msa["best_actual_running_time"])
    # ax.set(xlabel = '')
    # ax.set(ylabel='')
    # plt.show()


def grid_search_time_and_rf(rf_mod_err, rf_mod_time, data_dict, epsilon, show_plots=True):
    test_data = data_dict["full_test_data"][
        ["msa_name", "spr_radius", "spr_cutoff", "n_parsimony", "n_random", "running_time_vs_default",
         "default_mean_Err_normalized", "n_seq", "n_loci"
         ]].copy()
    test_data["predicted_errors"] = rf_mod_err.predict(data_dict["X_test"])
    test_data["predicted_times"] = rf_mod_time.predict(data_dict["X_test"])
    test_data["actual_errors"] = data_dict["y_test_err"]
    test_data["actual_times"] = data_dict["y_test_time"]
    if show_plots:
        ax = sns.scatterplot(x=test_data["actual_errors"],
                             y=test_data["predicted_errors"])
        plt.show()
    allowed_predicted_error_data = test_data[(test_data["predicted_errors"] > epsilon)]
    allowed_actual_error_data = test_data[(test_data["actual_errors"] > epsilon)]
    predicted_best_time_per_msa = pd.merge(allowed_predicted_error_data,
                                           allowed_predicted_error_data.groupby("msa_name").agg(
                                               best_predicted_running_time=(
                                                   'predicted_times', max)).reset_index(),
                                           left_on=["msa_name", "predicted_times"],
                                           right_on=["msa_name", "best_predicted_running_time"])
    actual_best_time_per_msa = pd.merge(allowed_actual_error_data,
                                        allowed_actual_error_data.groupby("msa_name").agg(
                                            best_actual_running_time=(
                                                'actual_times', max)).reset_index(),
                                        left_on=["msa_name", "actual_times"],
                                        right_on=["msa_name", "best_actual_running_time"])

    predicted_best_time_per_msa = predicted_best_time_per_msa.groupby("msa_name").first().reset_index()
    actual_best_time_per_msa = actual_best_time_per_msa.groupby("msa_name").first().reset_index()
    if show_plots:
        ax = sns.scatterplot(
            x=predicted_best_time_per_msa[predicted_best_time_per_msa["actual_errors"] > -0.0001]["actual_errors"],
            y=predicted_best_time_per_msa[predicted_best_time_per_msa["actual_errors"] > -0.0001]["predicted_errors"])
        plt.show()
    valid_predicted_results = predicted_best_time_per_msa[
        (predicted_best_time_per_msa["actual_errors"] > epsilon) & (predicted_best_time_per_msa["actual_times"] >= 1)][
        ["actual_times"]]

    valid_predicted_results["Type"] = "Obtained via ML"  # plt.show()
    valid_actual_results = actual_best_time_per_msa[
        (actual_best_time_per_msa["actual_errors"] > epsilon) & (actual_best_time_per_msa["actual_times"] >= 1)][
        ["actual_times"]]
    valid_actual_results["Type"] = "Potential"
    if show_plots:
        actual_vs_predicted = pd.concat([valid_predicted_results, valid_actual_results])
        sns.set(font_scale=1.5)
        ax = sns.histplot(x="actual_times", hue="Type", data=actual_vs_predicted)
        ax.set(xlabel='', ylabel='')
        plt.show()


def train_models(data_dict):
    logging.info("About to calculate RF for modelling error")
    rf_mod_err = classifier(data_dict["X_train"], data_dict["y_train_err"],
                            path=os.path.join(ML_RESULTS_FOLDER, "Err_rf_new"))
    err_var_impt = variable_importance(data_dict["X_train"], rf_mod_err)
    logging.info(f"Error RF variable importance: \n {err_var_impt}")
    y_test_err_predicted = rf_mod_err.predict(data_dict["X_test"])
    err_test_metrics = model_metrics(data_dict["y_test_err"], y_test_err_predicted, is_classification=True)
    logging.info(f"Error RF metrics: \n {err_test_metrics}")
    rf_mod_time = regressor(data_dict["X_train"], data_dict["y_train_time"],
                            path=os.path.join(ML_RESULTS_FOLDER, "time_rf_new"))
    time_var_impt = variable_importance(data_dict["X_train"], rf_mod_time)
    logging.info(f"Time RF variable importance: \n {time_var_impt}")
    y_test_time_predicted = rf_mod_time.predict(data_dict["X_test"])
    time_test_metrics = model_metrics(data_dict["y_test_time"], y_test_time_predicted, is_classification=False)
    logging.info(f"Time_test_metrics: \n {time_test_metrics}")
    return rf_mod_err, rf_mod_time


def split_to_train_and_test(full_data, data_feature_names, search_feature_names):
    train_data, test_data, validation_data = train_test_validation_splits(
        full_data, test_pct=0.3, val_pct=0)
    X_train = train_data[data_feature_names + search_feature_names]
    y_train_err = train_data["is_global_max"]
    y_train_time = train_data["normalized_time"]
    X_test = test_data[data_feature_names + search_feature_names]
    y_test_err = test_data["is_global_max"]
    y_test_time = test_data["normalized_time"]
    return {"X_train": X_train, "y_train_err": y_train_err, "y_train_time": y_train_time, "X_test": X_test,
            "y_test_err": y_test_err, "y_test_time": y_test_time, "full_test_data": test_data}


def main():
    epsilon = 0.1
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', action='store', type=str,
                        default=f"/Users/noa/Workspace/raxml_deep_learning_results/current_raw_results/global_csv_enriched_new.tsv")
    parser.add_argument('--features_path', action='store', type=str,
                        default=f"/Users/noa/Workspace/raxml_deep_learning_results/current_ML_results/features{CSV_SUFFIX}")

    args = parser.parse_args()
    features = pd.read_csv(args.features_path, sep=CSV_SEP)
    data = pd.read_csv(args.data_path, sep=CSV_SEP)
    data["is_global_max"] = data["delta_ll_from_overall_msa_best_topology"] < epsilon
    data["normalized_time"] = data["elapsed_running_time"] / data["test_norm_const"]
    features["msa_name"] = features["msa_path"].apply(lambda s: remove_env_path_prefix(s))
    data["msa_name"] = data["msa_path"].apply(lambda s: remove_env_path_prefix(s))
    data["starting_tree_bool"] = data["starting_tree_type"] == "pars"
    data["starting_tree_ll"] = \
    data.groupby(['msa_path', 'starting_tree_type']).transform(lambda x: (x - x.mean()) / x.std())["starting_tree_ll"]
    data["normalized_time"] = data.groupby('msa_path').transform(lambda x: (x - x.mean()) / x.std())["normalized_time"]
    full_data = data.merge(features, on="msa_name")
    # full_data = full_data.replace([np.inf, -np.inf,np.nan], -1)
    all_jobs_general_log_file = os.path.join(ML_RESULTS_FOLDER, "ML_log_file.log")
    create_dir_if_not_exists(ML_RESULTS_FOLDER)
    logging_level = logging.INFO
    logging.basicConfig(filename=all_jobs_general_log_file, level=logging_level)
    msa_features = [col for col in full_data.columns if col.startswith("feature_")]
    search_features = ['spr_radius', 'spr_cutoff', 'starting_tree_bool', "starting_tree_ll"]
    data_dict = split_to_train_and_test(full_data, msa_features, search_features)
    rf_mod_err, rf_mod_time = train_models(data_dict)

    # plot_full_data_metrics(full_data, epsilon)
    # grid_search_time_and_rf(rf_mod_err, rf_mod_time, data_dict, epsilon)


if __name__ == "__main__":
    main()
