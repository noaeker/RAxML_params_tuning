import sys
from sklearn.calibration import calibration_curve

if sys.platform == "linux" or sys.platform == "linux2":
    PROJECT_ROOT_DIRECRTORY = "/groups/pupko/noaeker/RAxML_params_tuning"
else:
    PROJECT_ROOT_DIRECRTORY = "/Users/noa/Workspace/RAxML_params_tuning"
sys.path.append(PROJECT_ROOT_DIRECRTORY)

from side_code.config import *
from ML_pipeline.side_functions import get_ML_parser
from ML_pipeline.ML_pipeline_procedures import get_average_results_on_default_configurations_per_msa,edit_raw_data_for_ML
from ML_pipeline.ML_algorithms_and_hueristics import ML_model, print_model_statistics, train_test_validation_splits, \
    variable_importance
from ML_pipeline.knapsack import knapsack_on_test_set
from ML_pipeline.multi_tree_model import get_multitree_performance_on_test_set
import pandas as pd
import os
import argparse
import numpy as np
import pickle



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
    return {"X_train": X_train, "train_MSAs": train_data["msa_path"], "y_train_err": y_train_err, "y_train_time": y_train_time, "X_test": X_test,
            "y_test_err": y_test_err, "y_test_time": y_test_time, "X_val": X_val, "y_val_err": y_val_err, "y_val_time": y_val_time,
            "full_test_data": test_data, "full_train_data" : train_data, "full_validation_data": validation_data}



def get_file_paths(args):
    return {"features_path": f"{args.baseline_folder}/all_features{CSV_SUFFIX}",
     "ML_edited_features_path": f"{args.baseline_folder}/ML_edited_features{CSV_SUFFIX}",
     "default_path": f"{args.baseline_folder}/default_sampling{CSV_SUFFIX}",
     "error_model_path": f"{args.baseline_folder}/error.model",
     "required_accuracy_model_path": f"{args.baseline_folder}/accuracy.model",
    "validation_multi_tree_data": f"{args.baseline_folder}/validation_multi_tree_data_dict",
            "test_multi_tree_data": f"{args.baseline_folder}/test_multi_tree_data_dict",
    "performance_on_test_set":f"{args.baseline_folder}/overall_performance_on_test_set{CSV_SUFFIX}",
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
    msa_features = [col for col in data_for_ML.columns if
                    (col.startswith("feature_")  and col not in ["feature_msa_path", "feature_msa_name",
                                                                "feature_msa_type","feature_msa_gaps"])]
    for col in msa_features:
        print(col)
        data_for_ML[col] = pd.to_numeric(data_for_ML[col])
    logging.info(f"Features are: {msa_features}")
    search_features = ['spr_radius', 'spr_cutoff', 'starting_tree_bool', "starting_tree_ll"]
    data_dict = get_ML_ready_data(data_for_ML, msa_features, search_features, test_pct=args.test_pct,
                                  val_pct=args.val_pct)
    return data_dict



def generate_single_tree_models(data_dict, file_paths, args):
    time_model= ML_model(X_train=data_dict["X_train"], groups =data_dict["train_MSAs"], y_train=data_dict["y_train_time"],n_jobs=args.n_jobs, path = file_paths["time_model_path"], classifier = False)
    print_model_statistics(model=time_model,  test_X=data_dict["X_test"],
                           y_test=data_dict["y_test_time"], is_classification=False, vi_path=file_paths["time_vi"],
                           name="Time regression model")
    error_model= ML_model(X_train= data_dict["X_train"], groups= data_dict["train_MSAs"], y_train= data_dict["y_train_err"],n_jobs=args.n_jobs,
                          path = file_paths["error_model_path"], classifier = True)
    print_model_statistics(model=error_model,  test_X=data_dict["X_test"],
                           y_test=data_dict["y_test_err"], is_classification=True, vi_path=file_paths["error_vi"],
                           name="Error classification model")

    return time_model,error_model


def train_multi_tree_models(file_paths, time_model, error_model, data_dict, total_MSA_level_features, args):
    if not os.path.exists(file_paths["performance_on_test_set"]):
        logging.info(f"Starting ML model from scratch {file_paths['performance_on_test_set']}")

        if args.tree_choosing_method=='knapsack':
            performance_on_test_set = knapsack_on_test_set(args.baseline_folder,data_dict,time_model,error_model)

        elif args.tree_choosing_method=='ML':
            performance_on_test_set =get_multitree_performance_on_test_set(data_dict,args,time_model,error_model,total_MSA_level_features,file_paths)


        performance_on_test_set.to_csv(file_paths["performance_on_test_set"], sep = CSV_SEP)
    else:
        logging.info(f"Using existing test performance in {file_paths['performance_on_test_set']}")
        performance_on_test_set = pd.read_csv(file_paths["performance_on_test_set"], sep = CSV_SEP)
    return performance_on_test_set



def main():
    epsilon = 0.1
    parser = get_ML_parser()
    args = parser.parse_args()
    file_paths = get_file_paths(args)
    features_data = pd.read_csv(file_paths["features_path"], sep=CSV_SEP)

    #msa_names = list(np.unique(features_data["msa_path"]))
    #np.random.seed(SEED)
    #chosen_msas= np.random.choice(msa_names, size=int(len(msa_names) * (0.2)), replace=False)
    #features_data = features_data[features_data["msa_path"].isin(chosen_msas)]
    logging_level = logging.INFO
    if os.path.exists(file_paths["log_file"]):
        os.remove(file_paths["log_file"])
    logging.basicConfig(filename=file_paths["log_file"], level=logging_level)
    features_data = features_data.loc[~features_data.msa_path.str.contains("single-gene_alignments")]


    logging.info(f"Enriching features data in {file_paths['features_path']} and saving to {file_paths['ML_edited_features_path']}")
    edited_data = edit_raw_data_for_ML(features_data, epsilon)
    enriched_features_data = edited_data["non_default"]
    enriched_default_data = edited_data["default"]
    #enriched_features_data.to_csv(file_paths["ML_edited_features_path"], sep=CSV_SEP)
    #enriched_default_data.to_csv(file_paths["ML_edited_default_data_path"], sep=CSV_SEP)

    data_dict = generate_basic_data_dict(enriched_features_data, args)
    time_model, error_model = generate_single_tree_models(data_dict, file_paths, args)


    total_MSA_level_features = edited_data["MSA_level_columns"]+edited_data["averaged_MSA_level_columns"]
    performance_on_test_set = train_multi_tree_models(file_paths, time_model, error_model, data_dict, total_MSA_level_features, args)
    if not os.path.exists(file_paths["default_path"]):
        logging.info(f"Generating default data from beggining")
        default_data_performance = get_average_results_on_default_configurations_per_msa(enriched_default_data[enriched_default_data.msa_path.isin(list(performance_on_test_set["msa_path"].unique()))],
                                                                                         n_sample_points=args.n_sample_points,
                                                                                         seed=SEED)
        default_data_performance.to_csv(file_paths["default_path"], sep=CSV_SEP)

    else:
        logging.info(f"Using existing default data in {file_paths['default_path']}")
        default_data_performance = pd.read_csv(file_paths["default_path"], sep=CSV_SEP)

    raw_comp = performance_on_test_set.merge(default_data_performance, how = 'left', on="msa_path")
    aggregated_comp =raw_comp.groupby(by=["msa_path","clusters_max_dist","max_accuracy","feature_msa_pypythia_msa_difficulty","predicted_total_accuracy_calibrated","predicted_total_accuracy","predicted_iid_single_tree_failure_probability","n_parsimony_trees_used","n_random_trees_used","status","diff","total_time_predicted","total_actual_time"]).agg(mean_default_diff = ('default_final_err',np.mean),mean_default_status = ('default_status', np.mean)).reset_index()
    #overall_results = aggregated_comp.groupby
    aggregated_comp.to_csv(file_paths["final_comparison_path"], sep=CSV_SEP)







if __name__ == "__main__":
    main()
