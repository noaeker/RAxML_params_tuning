from side_code.config import *
from side_code.file_handling import create_dir_if_not_exists
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import pickle
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, roc_auc_score
import numpy as np
import os
import argparse


# Mean absolute error (MAE)
def rf_metrics(y_test, predictions):
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


#
# def rf_classifier(X_train, y_train):
#     rf_err_file = "rf_classifier"
#     if os.path.exists(rf_err_file):
#         rf = pickle.load(open(rf_err_file, 'rb'))
#         # Train the model on training data
#     else:
#         rf = RandomForestClassifier()
#         rf.fit(X_train, y_train)
#         pickle.dump(rf, open(rf_err_file, 'wb'))
#         # Calculate the absolute errors
#     return rf


def rf_regressor(X_train, y_train, path):
    if os.path.exists(path):
        rf = pickle.load(open(path, 'rb'))
    else:
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        pickle.dump(rf, open(path, 'wb'))
        # Calculate the absolute errors
    return rf



def grid_search_time_and_rf(rf_mod_err, rf_mod_time, data_dict, epsilon):
    full_data = data_dict["full_test_data"][
        ["msa_name", "spr_radius", "spr_cutoff", "n_parsimony", "n_random", "running_time_vs_default",
         "default_mean_Err_normalized"
         ]].copy()
    full_data["predicted_errors"] = rf_mod_err.predict(data_dict["X_test"])
    full_data["predicted_times"] = rf_mod_time.predict(data_dict["X_test"])
    full_data["actual_errors"] = data_dict["y_test_err"]
    full_data["actual_times"] = data_dict["y_test_time"]
    allowed_predicted_error_data = full_data[(full_data["predicted_errors"] > epsilon)]
    allowed_actual_error_data = full_data[(full_data["actual_errors"] > epsilon)]
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
    #ax = sns.histplot(predicted_best_time_per_msa["predicted_errors"]-predicted_best_time_per_msa["actual_errors"])
    #plt.show()
    ax = sns.scatterplot(x = predicted_best_time_per_msa["actual_errors"], y = predicted_best_time_per_msa["actual_times"])
    plt.show()



def train_rf_models(data_dict):
    logging.info("About to calculate RF for modelling error")
    rf_mod_err = rf_regressor(data_dict["X_train"], data_dict["y_train_err"],
                              path=os.path.join(ML_RESULTS_FOLDER, "Err_rf_new"))
    err_var_impt = variable_importance(data_dict["X_train"], rf_mod_err)
    logging.info(f"Error RF variable importance: \n {err_var_impt}")
    y_test_err_predicted = rf_mod_err.predict(data_dict["X_test"])
    err_test_metrics = rf_metrics(data_dict["y_test_err"], y_test_err_predicted)
    logging.info(f"Error RF metrics: \n {err_test_metrics}")
    rf_mod_time = rf_regressor(data_dict["X_train"], data_dict["y_train_time"],
                               path=os.path.join(ML_RESULTS_FOLDER, "time_rf_new"))
    time_var_impt = variable_importance(data_dict["X_train"], rf_mod_time)
    logging.info(f"Time RF variable importance: \n {time_var_impt}")
    y_test_time_predicted = rf_mod_time.predict(data_dict["X_test"])
    time_test_metrics = rf_metrics(data_dict["y_test_time"], y_test_time_predicted)
    logging.info(f"Time_test_metrics: \n {time_test_metrics}")
    return rf_mod_err, rf_mod_time


def split_to_train_and_test(full_data, data_feature_names, search_feature_names):
    train_data, test_data, validation_data = train_test_validation_splits(
        full_data, test_pct=0.2, val_pct=0)
    X_train = train_data[data_feature_names + search_feature_names]
    y_train_err = train_data["mean_Err_normalized"]
    y_train_time = train_data["running_time_vs_default"]
    X_test = test_data[data_feature_names + search_feature_names]
    y_test_err = test_data["mean_Err_normalized"]
    y_test_time = test_data["running_time_vs_default"]
    return {"X_train": X_train, "y_train_err": y_train_err, "y_train_time": y_train_time, "X_test": X_test,
            "y_test_err": y_test_err, "y_test_time": y_test_time, "full_test_data": test_data}


def main():
    epsilon = -1 * (10 ** -4)
    parser = argparse.ArgumentParser()
    parser.add_argument('--ML_data_path', action='store', type=str,
                        default=f"{ML_RESULTS_FOLDER}/final_ML_dataset{CSV_SUFFIX}")
    args = parser.parse_args()
    all_jobs_general_log_file = os.path.join(ML_RESULTS_FOLDER, "ML_log_file.log")
    create_dir_if_not_exists(ML_RESULTS_FOLDER)
    logging_level = logging.INFO
    logging.basicConfig(filename=all_jobs_general_log_file, level=logging_level)
    data_features = ['n_seq', 'n_loci', 'avg_tree_divergence',
                     'avg_largest_branch_length', 'avg_largest_distance_between_taxa', 'avg_tree_MAD',
                     'msa_type_numeric',
                     'avg_parsimony_rf_dist', 'mean_unique_topolgies_rf_dist', 'max_parsimony_rf_dist',
                     'best_parsimony_vs_best_random', 'parsimony_ll_var_vs_random_ll_var']
    search_features = ['spr_radius', 'spr_cutoff', 'n_parsimony',
                       'n_random']

    full_data = pd.read_csv(args.ML_data_path, sep=CSV_SEP)
    #full_data["avg_parsimony_rf_dist"] = full_data["avg_parsimony_rf_dist"]/(2*(full_data["n_seq"]-3))
    data_dict = split_to_train_and_test(full_data, data_features, search_features)
    rf_mod_err, rf_mod_time = train_rf_models(data_dict)

    grid_search_time_and_rf(rf_mod_err, rf_mod_time, data_dict, epsilon)


if __name__ == "__main__":
    main()
