import sys
sys.append('../')
from side_code.help_functions import *
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import pickle
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, roc_auc_score



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




def train_error_model(train_data,data_features,search_features,test_data):
    X_train = train_data[data_features + search_features]
    y_train_err = train_data["mean_Err_normalized"]
    X_test = test_data[data_features + search_features]
    y_test_err = test_data["mean_Err_normalized"]
    rf_mod_err = rf_regressor(X_train, y_train_err, name="Err_rf_new")
    return rf_mod_err

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

def test_RF_model_predictions():
    1==1





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ML_data_path', action='store', type=str, default=f"{RESULTS_FOLDER}/full_raxml_data.tsv")
    args = parser.parse_args()
    data_features = ['n_seq', 'n_loci', 'parsimony_tree_alpha', 'tree_divergence',
                     'largest_branch_length', 'largest_distance_between_taxa', 'tree_MAD', 'msa_type_numeric',
                     'avg_parsimony_rf_dist', 'parsimony_vs_random_diff', 'parsimony_var_vs_mean', 'random_var_vs_mean','best_parsimony_vs_best_random','distances_vs_ll_corr']
    search_features = ['spr_radius', 'spr_cutoff', 'n_parsimony',
                       'n_random']

    output_test_path = f"test_data_for_greed_search_new{CSV_SUFFIX}"
    if os.path.exists(output_test_path):
        test = pd.read_csv(output_test_path, sep = CSV_SEP)
    else:
        test = train_rf_models(args.ML_data_path, data_features, search_features, output_test_path= output_test_path)


    grid_search_time_and_rf(test)

if __name__ == "__main__":
    main()
