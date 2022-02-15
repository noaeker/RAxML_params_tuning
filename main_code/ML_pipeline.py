from help_functions import *
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# Import the model we are using
from sklearn.ensemble import RandomForestRegressor
import pickle
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
# Instantiate model with 1000 decision trees


# Mean absolute error (MAE)
def rf_metrics(y_test, predictions):
    mae = mean_absolute_error(y_test.values.ravel(), predictions)

    # Mean squared error (MSE)
    mse = mean_squared_error(y_test.values.ravel(), predictions)

    # R-squared scores
    r2 = r2_score(y_test.values.ravel(), predictions)

    print('Mean Absolute Error:', round(mae, 2))
    print('Mean Squared Error:', round(mse, 2))
    print('R-squared scores:', round(r2, 2))


# Print metrics


def train_test_splits(full_data, features, label, test_pct):
    np.random.seed(SEED)
    msa_names = list(np.unique(full_data["msa_name"]))
    test_msas = np.random.choice(msa_names, size = int(len(msa_names)*test_pct), replace = False)
    train_data = full_data[~full_data["msa_name"].isin(test_msas)]
    test_data = full_data[full_data["msa_name"].isin(test_msas)]
    train_x = train_data[features]
    train_y = train_data[label]
    test_x = test_data[features]
    test_y = test_data[label]
    return train_x, train_y, test_x, test_y




def plot_results(x_axis,y_test,y_test_hat):
    # Build scatterplot
    err = y_test-y_test_hat
    # plt.scatter(x_axis, y_test, c='b', alpha=0.5, marker='.', label='Real')
    # plt.xlabel('n_seq')
    # plt.ylabel('Err')
    # plt.grid(color='#D3D3D3', linestyle='solid')
    # plt.legend(loc='lower right')
    # plt.show()

    plt.scatter(y_test_hat, y_test, c='b', alpha=0.5, marker='.', label='Real')
    plt.grid(color='#D3D3D3', linestyle='solid')
    plt.legend(loc='lower right')
    plt.show()



def summarize_results_per_msa(raw_data):
    data = raw_data[["msa_name","best_msa_ll"]]
    data = data.drop_duplicates()
    return data



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_data_path', action='store', type=str, default=f"{RESULTS_FOLDER}/full_raxml_data.tsv")
    parser.add_argument('--label_path', action='store', type=str, default=f"{RESULTS_FOLDER}/sampled_raxml_data_large.tsv")
    parser.add_argument('--features_path', action='store', type=str,
                        default=f"{RESULTS_FOLDER}/features{CSV_SUFFIX}")
    args = parser.parse_args()
    raw_data = pd.read_csv(args.raw_data_path, sep=CSV_SEP)
    per_msa_data = summarize_results_per_msa(raw_data)
    label = pd.read_csv(args.label_path, sep = CSV_SEP)
    label = label[label["spr_radius"]!="default"]
    label = pd.merge(label, per_msa_data, on = ["msa_name"])
    label["normalized_Err"] = label["mean_Err"]/ label["best_msa_ll"]
    features = pd.read_csv(args.features_path, sep=CSV_SEP)
    full_data= pd.merge(features,label, on = ["msa_name"])
    data_features  = ['n_seq', 'n_loci','parsimony_tree_alpha', 'tree_divergence',
       'largest_branch_length', 'largest_distance_between_taxa', 'tree_MAD']
    search_features =['spr_radius', 'spr_cutoff', 'n_parsimony',
       'n_random']
    X_train_err, y_train_err, X_test_err, y_test_err = train_test_splits(full_data, features=data_features+search_features, label = "normalized_Err", test_pct = 0.2)
    rf_err_file = "rf_err_tmp"
    if os.path.exists(rf_err_file):
        rf = pickle.load(open( rf_err_file, 'rb'))
        # Train the model on training data
    else:
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train_err, y_train_err)
        pickle.dump(rf, open( rf_err_file, 'wb'))
        # Calculate the absolute errors
    print("got here")
    test_predictions = rf.predict(X_test_err)
    #train_predictions = rf.predict(X_train_err)
    rf_metrics(y_test_err, test_predictions)
    #rf_metrics(y_train_err, train_predictions)
    plot_results(X_test_err["n_seq"], y_test_err , test_predictions )
    #plot_results(X_train_err["n_seq"], y_train_err, train_predictions)





if __name__ == "__main__":
    main()