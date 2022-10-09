import sys

if sys.platform == "linux" or sys.platform == "linux2":
    PROJECT_ROOT_DIRECRTORY = "/groups/pupko/noaeker/RAxML_params_tuning"
else:
    PROJECT_ROOT_DIRECRTORY = "/Users/noa/Workspace/RAxML_params_tuning"
sys.path.append(PROJECT_ROOT_DIRECRTORY)

from side_code.config import *
from ML_pipeline.ML_pipeline_procedures import get_MSA_clustering_and_threshold_results,get_average_results_on_default_configurations_per_msa,try_different_tree_selection_metodologies, edit_raw_data_for_ML, choose_best_tree_selection_algorithm
from ML_pipeline.ML_algorithms_and_hueristics import classifier, regressor, print_model_statistics, train_test_validation_splits, \
    variable_importance
import pandas as pd
import os
import argparse
import numpy as np




def get_default_data_performance(args, default_path,test_MSAs, features_data):
    if not os.path.exists(default_path):
        logging.info(f"Generating default data from beggining")
        default_data = features_data[
            (features_data["type"] == "default") & (features_data["msa_path"].isin(test_MSAs))]
        logging.info("Getting default data performance")
        default_data_performance = get_average_results_on_default_configurations_per_msa(default_data,
                                                                                         n_sample_points=args.n_sample_points,
                                                                                         seed=SEED)
        default_data_performance.to_csv(default_path, sep=CSV_SEP)

    else:
        logging.info(f"Using existing default data in {default_path}")
        default_data_performance = pd.read_csv(default_path, sep=CSV_SEP)
    return default_data_performance



def get_ML_ready_data(full_data, data_feature_names, search_feature_names, test_pct, val_pct):
    train_data, test_data, validation_data = train_test_validation_splits(
        full_data, test_pct=test_pct, val_pct= val_pct)
    X_train = train_data[data_feature_names + search_feature_names]
    y_train_err = train_data["is_global_max"]
    y_train_time = train_data["normalized_relative_time"]
    X_test = test_data[data_feature_names + search_feature_names]
    y_test_err = test_data["is_global_max"]
    y_test_time = test_data["normalized_relative_time"]
    X_val = validation_data[data_feature_names + search_feature_names]
    y_val_err = validation_data["is_global_max"]
    y_val_time = validation_data["normalized_relative_time"]
    return {"X_train": X_train, "y_train_err": y_train_err, "y_train_time": y_train_time, "X_test": X_test,
            "y_test_err": y_test_err, "y_test_time": y_test_time, "X_val": X_val, "y_val_err": y_val_err, "y_val_time": y_val_time,
            "full_test_data": test_data, "full_train_data" : train_data, "full_validation_data": validation_data}



def get_file_paths(args):
    return {"features_path": f"{args.baseline_folder}/all_features{CSV_SUFFIX}",
     "tree_pairs_path": f"{args.baseline_folder}/tree_comparisons{CSV_SUFFIX}",
     "ML_edited_features_path": f"{args.baseline_folder}/ML_edited_features{CSV_SUFFIX}",
     "default_path": f"{args.baseline_folder}/default_sampling{CSV_SUFFIX}",
     "val_performance_for_each_methodology": f"{args.baseline_folder}/val_methodology_performance{CSV_SUFFIX}", #trying different methodologies for each msa.
     "test_performance_for_each_methodology":f"{args.baseline_folder}/test_methodology_performance{CSV_SUFFIX}",
     "enriched_validation_data_path": f"{args.baseline_folder}/enriched_validation_data{CSV_SUFFIX}",
            "enriched_test_data_path": f"{args.baseline_folder}/enriched_test_data{CSV_SUFFIX}",
     "error_model_path": f"{args.baseline_folder}/error.model",
     "required_accuracy_model_path": f"{args.baseline_folder}/accuracy.model",
            "best_cluster_model_path": f"{args.baseline_folder}/best_cluster.model",
    "performance_on_test_set":f"{args.baseline_folder}/performance_on_test_set{CSV_SUFFIX}",
     "time_model_path": f"{args.baseline_folder}/time.model",
     "final_comparison_path": f"{args.baseline_folder}/final_performance_comp{CSV_SUFFIX}",
     "log_file": f"{args.baseline_folder}/ML_log_file.log","time_vi": f"{args.baseline_folder}/time_vi{CSV_SUFFIX}",
            "error_vi": f"{args.baseline_folder}/error_vi{CSV_SUFFIX}",
            "final_error_vi": f"{args.baseline_folder}/final_error_vi{CSV_SUFFIX}",
            "agg_validation_tree_selection_results": f"{args.baseline_folder}/agg_tree_selection{CSV_SUFFIX}",
            "ML_edited_default_data_path": f"{args.baseline_folder}/ML_edited_default_data{CSV_SUFFIX}"
            }



def generate_basic_data_dict(data_for_ML, args):
    logging.info("Removing columns with NA")
    data_for_ML = data_for_ML.dropna(axis=1)  # remove columns with NAs
    msa_features = [col for col in data_for_ML.columns if col in ["pypythia_msa_difficulty"] or
                    (col.startswith("feature_") and col not in ["feature_msa_path", "feature_msa_name",
                                                                "feature_msa_type"])]
    for col in msa_features:
        data_for_ML[col] = pd.to_numeric(data_for_ML[col])
    logging.info(f"Features are: {msa_features}")
    search_features = ['spr_radius', 'spr_cutoff', 'starting_tree_bool', "starting_tree_ll"]
    data_dict = get_ML_ready_data(data_for_ML, msa_features, search_features, test_pct=args.test_pct,
                                  val_pct=args.val_pct)
    return data_dict



def generate_single_tree_models(data_dict, file_paths, args):
    time_model = regressor(data_dict["X_train"], data_dict["y_train_time"], args.n_jobs, file_paths["time_model_path"],
                           args.lightgbm)
    print_model_statistics(model=time_model, train_data=data_dict["X_train"], test_data=data_dict["X_test"],
                           y_test=data_dict["y_test_time"], is_classification=False, vi_path=file_paths["time_vi"],
                           name="Time regression model")
    error_model = classifier(data_dict["X_train"], data_dict["y_train_err"], args.n_jobs,
                             file_paths["error_model_path"], args.lightgbm)
    print_model_statistics(model=error_model, train_data=data_dict["X_train"], test_data=data_dict["X_test"],
                           y_test=data_dict["y_test_err"], is_classification=True, vi_path=file_paths["error_vi"],
                           name="Error classification model")
    return time_model, error_model

def generate_tree_groups_model_on_validation_data(file_paths, data_dict, time_model, error_model, args):
    if os.path.exists(file_paths["val_performance_for_each_methodology"]):
        validation_data_algorithms_performance_df = pd.read_csv(file_paths["val_performance_for_each_methodology"], sep=CSV_SEP)
    else: # Try
        enriched_validation_data = data_dict["full_validation_data"]
        enriched_validation_data["predicted_time"] = time_model.predict(data_dict["X_val"])
        enriched_validation_data["predicted_failure_probabilities"] = error_model.predict_proba(data_dict["X_val"])[:,
                                                                      0]
        enriched_validation_data.to_csv(file_paths["enriched_validation_data_path"], sep=CSV_SEP)

        validation_data_algorithms_performance_df = try_different_tree_selection_metodologies(args.baseline_folder,
                                                                                            enriched_validation_data,max_starting_trees= args.max_starting_trees,clusters_max_dist_options = args.clusters_max_dist_options)
        validation_data_algorithms_performance_df.to_csv(file_paths["val_performance_for_each_methodology"], sep=CSV_SEP)

    final_ML_features = [col for col in validation_data_algorithms_performance_df.columns if col.endswith('_mean') and col.startswith('feature')] + [
        "clusters_max_dist", "sum_of_predicted_success_probability", "total_time_predicted","n_random_trees_used","n_parsimony_trees_used"]
    final_ML_validation_data = validation_data_algorithms_performance_df[final_ML_features]
    logging.info("About to generate a final error model trained on validation data")
    total_error_model = classifier(final_ML_validation_data, validation_data_algorithms_performance_df["status"], 1,
                       file_paths["required_accuracy_model_path"])
    return final_ML_validation_data,total_error_model


def apply_tree_groups_model_on_test_data(validation_data_algorithms_performance_df,file_paths, data_dict, time_model, error_model, total_error_model, args):

    if os.path.exists(file_paths["test_performance_for_each_methodology"]):
       test_data_algorithms_performance_df = pd.read_csv(file_paths["test_performance_for_each_methodology"], sep=CSV_SEP)
    else: # Try
        enriched_test_data = data_dict["full_test_data"]

        enriched_test_data["predicted_time"] = time_model.predict(data_dict["X_test"])
        enriched_test_data["predicted_failure_probabilities"] = error_model.predict_proba(data_dict["X_test"])[:, 0]

        test_data_algorithms_performance_df = try_different_tree_selection_metodologies(args.baseline_folder,
                                                                                      enriched_test_data,
                                                                                      max_starting_trees= args.max_starting_trees,
                                                                                      clusters_max_dist_options=args.clusters_max_dist_options)

        test_data_algorithms_performance_df.to_csv(file_paths["test_performance_for_each_methodology"], sep=CSV_SEP)


    test_data_for_final_ML = test_data_algorithms_performance_df[[col for col in test_data_algorithms_performance_df.columns if col.endswith('_mean') and col.startswith('feature')]+["clusters_max_dist","sum_of_predicted_success_probability","total_time_predicted","n_random_trees_used","n_parsimony_trees_used"]]
    test_data_algorithms_performance_df['predicted_total_accuracy'] = total_error_model.predict_proba(test_data_for_final_ML)[:, 1]
    print_model_statistics(model=total_error_model,train_data=validation_data_algorithms_performance_df, test_data=test_data_for_final_ML, y_test=test_data_algorithms_performance_df["status"],vi_path= file_paths["final_error_vi"],
                           is_classification=True, name="Final Error classification model")
    test_data_algorithms_performance_df["max_accuracy"] = test_data_algorithms_performance_df.groupby("msa_path")['predicted_total_accuracy'].transform(np.max)
    test_performance_df= test_data_algorithms_performance_df[test_data_algorithms_performance_df.predicted_total_accuracy>=test_data_algorithms_performance_df.max_accuracy*0.7]
    test_performance_df = test_performance_df.sort_values("total_time_predicted").groupby('msa_path').head(1)
    test_performance_df.to_csv(file_paths["performance_on_test_set"], sep=CSV_SEP)
    return test_performance_df



def main():
    epsilon = 0.1
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline_folder', action='store', type=str, default=f"{READY_RAW_DATA}/all_data")
    parser.add_argument('--n_sample_points', action='store', type=int,
                        default=100)
    parser.add_argument('--val_pct', action='store', type=int,
                        default=0.25)
    parser.add_argument('--test_pct', action='store', type=int,
                        default=0.25)
    parser.add_argument('--max_starting_trees', action='store', type=int,
                        default=40)
    parser.add_argument('--clusters_max_dist_options', action='store', type=int,
                        default=np.linspace(0,1,5))
    parser.add_argument('--n_jobs', action='store', type=int,
                        default=1)
    parser.add_argument('--lightgbm', action='store_true', default=True)
    args = parser.parse_args()
    file_paths = get_file_paths(args)
    features_data = pd.read_csv(file_paths["features_path"], sep=CSV_SEP)
    logging_level = logging.INFO
    if os.path.exists(file_paths["log_file"]):
        os.remove(file_paths["log_file"])
    logging.basicConfig(filename=file_paths["log_file"], level=logging_level)
    features_data = features_data.loc[~features_data.msa_path.str.contains("single-gene_alignments")]
    if os.path.exists(file_paths["ML_edited_features_path"]) and os.path.exists(file_paths["ML_edited_default_data_path"]):
        logging.info(f"Using existing enriched features data in {file_paths['ML_edited_features_path']}")
        enriched_features_data = pd.read_csv(file_paths["ML_edited_features_path"], sep=CSV_SEP)
        enriched_default_data = pd.read_csv(file_paths["ML_edited_default_data_path"], sep=CSV_SEP)
    else:
        logging.info(f"Enriching features data in {file_paths['features_path']} and saving to {file_paths['ML_edited_features_path']}")
        edited_data = edit_raw_data_for_ML(features_data, epsilon)
        enriched_features_data = edited_data["non_default"]
        enriched_default_data = edited_data["default"]
        enriched_features_data.to_csv(file_paths["ML_edited_features_path"], sep=CSV_SEP)
        enriched_default_data.to_csv(file_paths["ML_edited_default_data_path"], sep=CSV_SEP)

    if not os.path.exists(file_paths["performance_on_test_set"]):
        logging.info(f"Starting ML model from scratch {file_paths['performance_on_test_set']}")
        data_dict = generate_basic_data_dict(enriched_features_data, args)
        time_model, error_model = generate_single_tree_models(data_dict, file_paths, args)
        final_ML_validation_data,total_error_model = generate_tree_groups_model_on_validation_data(file_paths, data_dict, time_model, error_model, args)
        performance_on_test_set= apply_tree_groups_model_on_test_data(final_ML_validation_data,file_paths, data_dict, time_model, error_model, total_error_model, args)
    else:
        logging.info(f"Using existing test performance in {file_paths['performance_on_test_set']}")
        performance_on_test_set = pd.read_csv(file_paths["performance_on_test_set"], sep = CSV_SEP)


    default_data_performance = get_default_data_performance(args,file_paths["default_path"], list(performance_on_test_set["msa_path"].unique()), enriched_default_data)
    #aggregated_default_results = default_data_performance.groupby(by=["msa_path"]).mean().reset_index()

    raw_comp = performance_on_test_set.merge(default_data_performance, how = 'left', on="msa_path")
    aggregated_comp =raw_comp.groupby(by=["msa_path","clusters_max_dist","n_parsimony_trees_used","n_random_trees_used","sum_of_predicted_success_probability","status","diff","total_time_predicted","total_actual_time"]).agg(mean_default_diff = ('default_final_err',np.mean),mean_default_status = ('default_status', np.mean)).reset_index()

    aggregated_comp.to_csv(file_paths["final_comparison_path"], sep=CSV_SEP)







if __name__ == "__main__":
    main()
