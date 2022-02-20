from help_functions import *
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# Import the model we are using
from sklearn.ensemble import RandomForestRegressor
import pickle
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid

from sklearn.model_selection import RandomizedSearchCV


# Instantiate model with 1000 decision trees


# Mean absolute error (MAE)
def rf_metrics(y_test, predictions):
    mae = mean_absolute_error(y_test, predictions)

    # Mean squared error (MSE)
    mse = mean_squared_error(y_test, predictions)

    # R-squared scores
    r2 = r2_score(y_test, predictions)

    print('Mean Absolute Error:', round(mae, 2))
    print('Mean Squared Error:', round(mse, 2))
    print('R-squared scores:', round(r2, 2))


# Print metrics


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


def plot_predicted_vs_actual(y_test, y_test_hat):
    plt.scatter(y_test_hat, y_test, c='b', alpha=0.5, marker='.', label='Real')
    plt.grid(color='#D3D3D3', linestyle='solid')
    plt.legend(loc='lower right')
    plt.show()


def summarize_results_per_msa(raw_data):
    data = raw_data[["msa_name", "best_msa_ll", "msa_type"]]
    data["msa_type_numeric"] = data["msa_type"] == "AA"
    data = data.drop_duplicates()
    return data


def enrich_sampling_data(sampling_data, raw_data):
    per_msa_data = summarize_results_per_msa(raw_data)
    sampling_data = pd.merge(sampling_data, per_msa_data, on=["msa_name"])
    sampling_data["is_default_run"] = (sampling_data["run_name"] == "default") & (
                sampling_data["n_parsimony"] == 10) & (sampling_data["n_random"] == 10)
    sampling_data["mean_Err_normalized"] = -sampling_data["mean_Err"] / sampling_data["best_msa_ll"]
    default_confg = sampling_data[sampling_data["is_default_run"]].copy()
    default_confg = default_confg.rename(
        columns={'mean_time': 'default_time', "mean_Err_normalized": "default_Err_normalized",
                 "mean_Err": "default_Err"})[
        ["msa_name", "default_time", "default_Err_normalized", "default_Err"]]
    enriched_sampling_data = pd.merge(sampling_data, default_confg, on=["msa_name"])
    enriched_sampling_data["normalized_error_vs_default"] = enriched_sampling_data["mean_Err_normalized"] - \
                                                            enriched_sampling_data["default_Err_normalized"]
    enriched_sampling_data["running_time_vs_default"] = enriched_sampling_data["default_time"] / enriched_sampling_data[
        "mean_time"]
    enriched_sampling_data = enriched_sampling_data[enriched_sampling_data["spr_radius"] != "default"]
    best_running_time = enriched_sampling_data[enriched_sampling_data["normalized_error_vs_default"] <= 0].groupby(
        ["msa_name"]).agg({'running_time_vs_default': 'max'}).rename(
        columns={"running_time_vs_default": "best_running_time_vs_default"})
    enriched_sampling_data = pd.merge(enriched_sampling_data, best_running_time, on=["msa_name"])

    enriched_sampling_data["is_optimal_run"] = (enriched_sampling_data["running_time_vs_default"]==enriched_sampling_data["best_running_time_vs_default"]) & (enriched_sampling_data["normalized_error_vs_default"] <= 0 )
    return enriched_sampling_data


# def perf_measure(y_actual, y_hat):
#     TP = 0
#     FP = 0
#     TN = 0
#     FN = 0
#
#     for i in range(len(y_hat)):
#         if y_actual[i] == y_hat[i] == True:
#             TP += 1
#         if y_hat[i] == True and y_actual[i] != y_hat[i]:
#             FP += 1
#         if y_actual[i] == y_hat[i] == False:
#             TN += 1
#         if y_hat[i] == False and y_actual[i] != y_hat[i]:
#             FN += 1
#
#     return {"TP": TP, "FP": FP, "TN": TN, "FN": FN}


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


def rf_regressor(X_train, y_train, name):
    rf_err_file = name
    if os.path.exists(rf_err_file):
        rf = pickle.load(open(rf_err_file, 'rb'))
    else:
        rf = RandomForestRegressor(n_estimators=100, random_state=42, oob_score=True)
        rf.fit(X_train, y_train)
        pickle.dump(rf, open(rf_err_file, 'wb'))
        # Calculate the absolute errors
    return rf



def get_predicted_outcome_per_test_msa(test_data, msa):
    msa_data = test_data[test_data["msa_name"] == msa]
    best_predicted_accuracy_msa_data = msa_data[msa_data['predicted_Err'] == msa_data['best_predicted_error']]
    best_predicted_accuracy = (min(best_predicted_accuracy_msa_data["predicted_Err"]))
    best_predicted_running_time = (max(best_predicted_accuracy_msa_data["predicted_time"]))
    corresponding_row = msa_data[(msa_data["predicted_Err"] == best_predicted_accuracy) & (
                msa_data["predicted_time"] == best_predicted_running_time)]
    running_time = min(corresponding_row["running_time_vs_default"])
    accurcy = (max(corresponding_row["mean_Err_normalized"]))
    return {"predicted_running_time":running_time, "predicted_accuracy":accurcy}

def grid_search_time_and_rf(test_data):
    potential_results_vs_default = test_data[test_data["is_optimal_run"]]["running_time_vs_default","msa_name","normalized_error_vs_default"]

    predicted_config_vs_default = [get_predicted_outcome_per_test_msa(test_data, msa) for msa in msas]
    predicted_accuracies = [predicted_accuracy_and_running_time[i]["predicted_accuracy"] for i in range(len(predicted_accuracy_and_running_time))]
    predicted_running_times = [predicted_accuracy_and_running_time[i]["predicted_running_time"] for i in
                          range(len(predicted_accuracy_and_running_time))]

    #sns.histplot(y= predicted_accuracies , color="red", bins=50)
    #sns.histplot(y= predicted_running_times, color="blue", bins=50)
    plt.show()
    print(np.mean(predicted_running_times))
    print(np.median(predicted_running_times))

    print(np.mean(potential_results_vs_default))
    print(np.median(potential_results_vs_default))
    #plt.show()




def train_rf_models(full_data, data_features, search_features,output_test_path):
    train_data, test_data, validation_data = train_test_validation_splits(
        full_data, test_pct=0.2, val_pct=0)
    X_train = train_data[data_features + search_features]
    y_train_err = train_data["mean_Err_normalized"]
    y_train_time = train_data["running_time_vs_default"]
    X_test = test_data[data_features + search_features]
    y_test_err = test_data["mean_Err_normalized"]
    y_test_time = test_data["running_time_vs_default"]
    rf_mod_err = rf_regressor(X_train, y_train_err, name="Err_rf_new")
    print(variable_importance(X_train, rf_mod_err))
    y_test_err_predicted = rf_mod_err.predict(X_test)
    rf_metrics(y_test_err, y_test_err_predicted)
    rf_mod_time = rf_regressor(X_train, y_train_time, name="time_rf_new")
    print(variable_importance(X_train, rf_mod_time))
    y_test_time_predicted = rf_mod_time.predict(X_test)
    rf_metrics(list(y_test_time), list(y_test_time_predicted))
    test_data["predicted_Err"] = y_test_err
    test_data["predicted_time"] = y_test_time_predicted
    test_data.to_csv(output_test_path, sep = CSV_SEP)
    return test_data




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_data_path', action='store', type=str, default=f"{RESULTS_FOLDER}/full_raxml_data.tsv")
    parser.add_argument('--label_path', action='store', type=str,
                        default=f"{RESULTS_FOLDER}/sampled_raxml_data_large.tsv")
    parser.add_argument('--features_path', action='store', type=str,
                        default=f"{RESULTS_FOLDER}/features{CSV_SUFFIX}")
    args = parser.parse_args()
    raw_data = pd.read_csv(args.raw_data_path, sep=CSV_SEP)
    sampling_data_label = pd.read_csv(args.label_path, sep=CSV_SEP)
    sampling_data_label = enrich_sampling_data(sampling_data_label, raw_data)
    msa_and_tree_features = pd.read_csv(args.features_path, sep=CSV_SEP)
    full_data = pd.merge(sampling_data_label,msa_and_tree_features, on=["msa_name"])
    data_features = ['n_seq', 'n_loci', 'parsimony_tree_alpha', 'tree_divergence',
                     'largest_branch_length', 'largest_distance_between_taxa', 'tree_MAD', 'msa_type_numeric',
                     'avg_parsimony_rf_dist', 'parsimony_vs_random_diff', 'parsimony_var_vs_mean', 'random_var_vs_mean','best_parsimony_vs_best_random','distances_vs_ll_corr']
    search_features = ['spr_radius', 'spr_cutoff', 'n_parsimony',
                       'n_random']

    output_test_path = f"test_data_for_greed_search_new{CSV_SUFFIX}"
    if os.path.exists(output_test_path):
        test = pd.read_csv(output_test_path, sep = CSV_SEP)
    else:
        test = train_rf_models(full_data, data_features, search_features, output_test_path= output_test_path)


    grid_search_time_and_rf(test)

if __name__ == "__main__":
    main()
