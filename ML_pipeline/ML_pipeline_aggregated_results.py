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


def train_models(data_dict):
    logging.info("About to calculate RF for modelling error")
    rf_mod_err = regressor(data_dict["X_train"], data_dict["y_train_err"],
                            path=os.path.join(ML_RESULTS_FOLDER, "Err_rf_new"))
    err_var_impt = variable_importance(data_dict["X_train"], rf_mod_err)
    logging.info(f"Error RF variable importance: \n {err_var_impt}")
    y_test_err_predicted = rf_mod_err.predict(data_dict["X_test"])
    err_test_metrics = model_metrics(data_dict["y_test_err"], y_test_err_predicted, is_classification=False)
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
        full_data, test_pct=0.1, val_pct=0)
    X_train = train_data[data_feature_names + search_feature_names]
    y_train_err = train_data["is_global_max"]
    y_train_time = train_data["normalized_time"]
    X_test = test_data[data_feature_names + search_feature_names]
    y_test_err = test_data["is_global_max"]
    y_test_time = test_data["normalized_time"]
    return {"X_train": X_train, "y_train_err": y_train_err, "y_train_time": y_train_time, "X_test": X_test,
            "y_test_err": y_test_err, "y_test_time": y_test_time, "full_test_data": test_data}


def edit_data(data, epsilon):
    data["is_global_max"] = data["delta_ll_from_overall_msa_best_topology"] <= epsilon
    data["relative_time"] = data["elapsed_running_time"] / data["test_norm_const"]
    data["msa_name"] = data["msa_path"].apply(lambda s: remove_env_path_prefix(s))
    data["starting_tree_bool"] = data["starting_tree_type"] == "pars"
    data["feature_tree_divergence_sclaed"] = data["feature_tree_divergence"]/ data["feature_n_seq"]


def edit_aggregated_data(data):
    data["normalized_time"] = data.groupby('msa_path').transform(lambda x: (x - x.mean()) / x.std())["relative_time"]
    idx = data.groupby(['msa_path'])['is_global_max'].transform(max) == data['is_global_max']
    data = data[idx]
    return data
    #data["normalized_global_max"] = data.groupby('msa_path').transform(lambda x: (x - x.mean()) / x.std())["is_global_max"]

def main():
    epsilon = 1
    parser = argparse.ArgumentParser()
    parser.add_argument('--features_path', action='store', type=str,
                        default=f"/Users/noa/Workspace/raxml_deep_learning_results/c_30_70/features.tsv")

    args = parser.parse_args()
    data = pd.read_csv(args.features_path, sep=CSV_SEP)
    edit_data(data, epsilon)
    data = data.groupby(['msa_name','msa_path','spr_radius', 'spr_cutoff', 'starting_tree_bool']).mean().reset_index()
    data = edit_aggregated_data(data)
    # full_data = full_data.replace([np.inf, -np.inf,np.nan], -1)
    all_jobs_general_log_file = os.path.join(ML_RESULTS_FOLDER, "ML_log_file.log")
    create_dir_if_not_exists(ML_RESULTS_FOLDER)
    logging_level = logging.INFO
    logging.basicConfig(filename=all_jobs_general_log_file, level=logging_level)
    msa_features = [col for col in data.columns if col.startswith("feature_") and col not in ["feature_msa_path", "feature_msa_name","feature_msa_type"]]
    search_features = ['spr_radius', 'spr_cutoff', 'starting_tree_bool', "starting_tree_ll"]

    data_dict = split_to_train_and_test(data, msa_features, search_features)
    rf_mod_err, rf_mod_time = train_models(data_dict)

    # plot_full_data_metrics(full_data, epsilon)
    # grid_search_time_and_rf(rf_mod_err, rf_mod_time, data_dict, epsilon)


if __name__ == "__main__":
    main()
