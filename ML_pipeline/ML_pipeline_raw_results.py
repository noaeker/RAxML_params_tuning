import sys

if sys.platform == "linux" or sys.platform == "linux2":
    PROJECT_ROOT_DIRECRTORY = "/groups/pupko/noaeker/RAxML_params_tuning"
else:
    PROJECT_ROOT_DIRECRTORY = "/Users/noa/Workspace/RAxML_params_tuning"
sys.path.append(PROJECT_ROOT_DIRECRTORY)

from side_code.config import *
from ML_pipeline.side_functions import get_ML_parser
from ML_pipeline.ML_pipeline_procedures import get_average_results_on_default_configurations_per_msa,edit_raw_data_for_ML
from ML_pipeline.ML_algorithms_and_hueristics import ML_model, print_model_statistics, train_test_validation_splits
import pandas as pd
import os
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
            'test_MSAs': test_data["msa_path"],"y_test_err": y_test_err, "y_test_time": y_test_time, "X_val": X_val, "y_val_err": y_val_err, "y_val_time": y_val_time,
            "full_test_data": test_data, "full_train_data" : train_data, "full_validation_data": validation_data}



def get_file_paths(args):
    return {"features_path": f"{args.baseline_folder}/all_features{CSV_SUFFIX}",
     "ML_edited_features_path": f"{args.baseline_folder}/ML_edited_features{CSV_SUFFIX}",
            "edited_data": f"{args.baseline_folder}/edited_data",
     "default_path": f"{args.baseline_folder}/default_sampling{CSV_SUFFIX}",
            "default_by_params_path": f"{args.baseline_folder}/default_by_params_sampling{CSV_SUFFIX}",
     "error_model_path": f"{args.baseline_folder}/error.model",
     "required_accuracy_model_path": f"{args.baseline_folder}/accuracy.model",
    "validation_multi_tree_data": f"{args.baseline_folder}/validation_multi_tree_data{CSV_SUFFIX}",
            "test_multi_tree_data": f"{args.baseline_folder}/test_multi_tree_data{CSV_SUFFIX}",
            "test_multi_tree_data_with_predictions": f"{args.baseline_folder}/test_multi_tree_data_with_predictions{CSV_SUFFIX}",
            "validation_single_tree_data": f"{args.baseline_folder}/validation_single_tree_data{CSV_SUFFIX}",
            "test_single_tree_data": f"{args.baseline_folder}/test_single_tree_data{CSV_SUFFIX}",
    "performance_on_test_set":f"{args.baseline_folder}/overall_performance_on_test_set{CSV_SUFFIX}",
     "time_model_path": f"{args.baseline_folder}/time.model",
     "final_comparison_path": f"{args.baseline_folder}/final_performance_comp{CSV_SUFFIX}",
            "final_comparison_path_agg": f"{args.baseline_folder}/final_performance_comp_agg{CSV_SUFFIX}",
     "log_file": f"{args.baseline_folder}/ML_log_file.log","time_vi": f"{args.baseline_folder}/time_vi{CSV_SUFFIX}",
            "error_vi": f"{args.baseline_folder}/error_vi{CSV_SUFFIX}",
            "final_error_vi": f"{args.baseline_folder}/final_error_vi{CSV_SUFFIX}",
            "ML_edited_default_data_path": f"{args.baseline_folder}/ML_edited_default_data{CSV_SUFFIX}"
            }



def generate_basic_data_dict(data_for_ML, args):
    #logging.info("Removing columns with NA")
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
    print_model_statistics(model=time_model,  test_X=data_dict["X_test"], test_groups = data_dict["test_MSAs"],
                           y_test=data_dict["y_test_time"], is_classification=False, vi_path=file_paths["time_vi"],
                           name="Time regression model")
    error_model= ML_model(X_train= data_dict["X_train"], groups= data_dict["train_MSAs"], y_train= data_dict["y_train_err"],n_jobs=args.n_jobs,
                          path = file_paths["error_model_path"], classifier = True)
    print_model_statistics(model=error_model,  test_X=data_dict["X_test"], test_groups = data_dict["test_MSAs"],
                           y_test=data_dict["y_test_err"], is_classification=True, vi_path=file_paths["error_vi"],
                           name="Error classification model")

    return time_model,error_model



def apply_single_tree_models_on_data(full_data,X, time_model,error_model,singletree_out_path):
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
        full_data.to_csv(singletree_out_path, sep=CSV_SEP)
        return full_data


# def train_multi_tree_models(file_paths, time_model, error_model, data_dict, total_MSA_level_features, args):
#     if not os.path.exists(file_paths["performance_on_test_set"]):
#         logging.info(f"Starting ML model from scratch {file_paths['performance_on_test_set']}")
#         performance_on_test_set =get_multitree_performance_on_test_set_per_threshold(data_dict, args, time_model, error_model, total_MSA_level_features, file_paths)
#
#
#         performance_on_test_set.to_csv(file_paths["performance_on_test_set"], sep = CSV_SEP)
#     else:
#         logging.info(f"Using existing test performance in {file_paths['performance_on_test_set']}")
#         performance_on_test_set = pd.read_csv(file_paths["performance_on_test_set"], sep = CSV_SEP)
#     return performance_on_test_set


def get_default_performance(enriched_default_data,args,performance_on_test_set, out_path):
    if not os.path.exists(out_path):
        logging.info(f"Generating default data from beggining")
        default_data_performance = get_average_results_on_default_configurations_per_msa(enriched_default_data[
                                                                                             enriched_default_data.msa_path.isin(
                                                                                                 list(
                                                                                                     performance_on_test_set[
                                                                                                         "msa_path"].unique()))],
                                                                                         n_sample_points=args.n_sample_points,
                                                                                         seed=SEED)
        default_data_performance.to_csv(out_path, sep=CSV_SEP)

    else:
        logging.info(f"Using existing default data in {out_path}")
        default_data_performance = pd.read_csv(out_path, sep=CSV_SEP)

    return default_data_performance



def main():
    epsilon = 0.1
    parser = get_ML_parser()
    args = parser.parse_args()
    file_paths = get_file_paths(args)
    features_data = pd.read_csv(file_paths["features_path"], sep=CSV_SEP)
    features_data = features_data.rename({'starting_tree_object_x':'starting_tree_object'})
    logging_level = logging.INFO
    if os.path.exists(file_paths["log_file"]):
        os.remove(file_paths["log_file"])
    logging.basicConfig(filename=file_paths["log_file"], level=logging_level)



    features_data = features_data.loc[~features_data.msa_path.str.contains("single-gene_alignments")]


    embedding_features = [col for col in  features_data.columns if ('iso' in col or 'mds' in col or 'lle' in col or 'TSNE' in col or 'PCA' in col) ]
    excluded_features = [col for col in embedding_features if 'time'  in col or 'lle' in col or 'TSNE' in col]
    #mds_included_features = [f'feature_mds_False_pca_{i}_3_spr_enriched' for i in range(3)]+['feature_mds_False_stress_3_spr_enriched']
    #for f in mds_included_features:
    #    features_data[f] = abs(features_data[f])
    mds_included_features = []

    # msa_names = list(np.unique(features_data["msa_path"]))
    # np.random.seed(SEED)
    # chosen_msas= np.random.choice(msa_names, size=int(len(msa_names) * (0.2)), replace=False)
    # features_data = features_data[features_data["msa_path"].isin(chosen_msas)]

    if os.path.exists(file_paths["edited_data"]):
        edited_data = pickle.load(open(file_paths["edited_data"],'rb'))
    else:
        features_data = features_data[[col for col in features_data.columns if col not in excluded_features]]
        #features_data = features_data.loc[features_data.starting_tree_type == 'rand']
        logging.info(f"Number of MSAs in feature data is {len(features_data['msa_path'].unique())}")

        logging.info(f"Enriching features data in {file_paths['features_path']} and saving to {file_paths['ML_edited_features_path']}")
        edited_data = edit_raw_data_for_ML(features_data, epsilon)
        with open(file_paths["edited_data"],"wb") as EDITED_DATA:
            pickle.dump(edited_data,EDITED_DATA)
        enriched_features_data = edited_data["non_default"]
        logging.info(f"Number of positive samples: {len(enriched_features_data.loc[enriched_features_data.is_global_max==1].index)}, Number of negative samples {len(enriched_features_data.loc[enriched_features_data.is_global_max==0].index)}")
        #enriched_default_data = edited_data["default"]
        #enriched_default_data_by_params = edited_data["default_by_params"]
        enriched_features_data.to_csv(file_paths["ML_edited_features_path"], sep=CSV_SEP)
        #enriched_default_data.to_csv(file_paths["ML_edited_default_data_path"], sep=CSV_SEP)

    data_dict = generate_basic_data_dict(edited_data["non_default"], args)
    time_model, error_model = generate_single_tree_models(data_dict, file_paths, args)

    #validation_data = apply_single_tree_models_on_data(data_dict["full_validation_data"], data_dict["X_val"],time_model, error_model,  file_paths["validation_single_tree_data"])
    test_data = apply_single_tree_models_on_data(data_dict["full_test_data"], data_dict["X_test"], time_model, error_model,
                                     file_paths["test_single_tree_data"])

    #default_data_performance = get_default_performance(edited_data["default"],args,performance_on_test_set, out_path = file_paths["default_path"])
    default_by_params_data_performance = get_default_performance(edited_data["default_by_params"], args, data_dict["full_test_data"], out_path= file_paths["default_by_params_path"])









if __name__ == "__main__":
    main()
