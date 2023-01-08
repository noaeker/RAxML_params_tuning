import sys

if sys.platform == "linux" or sys.platform == "linux2":
    PROJECT_ROOT_DIRECRTORY = "/groups/pupko/noaeker/RAxML_params_tuning"
else:
    PROJECT_ROOT_DIRECRTORY = "/Users/noa/Workspace/RAxML_params_tuning"
sys.path.append(PROJECT_ROOT_DIRECRTORY)

from side_code.config import *
from side_code.file_handling import create_dir_if_not_exists
from ML_pipeline.side_functions import get_ML_parser
from ML_pipeline.ML_pipeline_procedures import get_average_results_on_default_configurations_per_msa,edit_raw_data_for_ML
from ML_pipeline.ML_algorithms_and_hueristics import ML_model, print_model_statistics, train_test_validation_splits
import pandas as pd
import os
import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def get_ML_ready_data(full_data, data_feature_names, search_feature_names, test_pct, val_pct, subsample_train = False, subsample_train_frac = -1):
    train_data, test_data,validation_data = train_test_validation_splits(
        full_data, test_pct=test_pct, val_pct= val_pct, subsample_train = subsample_train, subsample_train_frac = subsample_train_frac)
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
            'test_MSAs': test_data["msa_path"],"y_test_err": y_test_err, "y_test_time": y_test_time,
            "full_test_data": test_data, "full_train_data" : train_data, "X_val": X_val, "y_val_err": y_val_err,"full_validation_data": validation_data,"y_val_time": y_val_time }



def get_file_paths(args):
    new_folder = os.path.join(args.baseline_folder, args.name)
    create_dir_if_not_exists(new_folder)
    return {"features_path": f"{args.baseline_folder}/all_features{CSV_SUFFIX}",
     "ML_edited_features_path": f"{new_folder}/ML_edited_features{CSV_SUFFIX}",
            "edited_data": f"{new_folder}/edited_data",
     "default_path": f"{new_folder}/default_sampling{CSV_SUFFIX}",
            "default_by_params_path": f"{new_folder}/default_by_params_sampling{CSV_SUFFIX}",
     "error_model_path": f"{new_folder}/error.model",
     "required_accuracy_model_path": f"{new_folder}/accuracy.model",
    "validation_multi_tree_data": f"{new_folder}/validation_multi_tree_data{CSV_SUFFIX}",
            "test_multi_tree_data": f"{new_folder}/test_multi_tree_data{CSV_SUFFIX}",
            "test_multi_tree_data_with_predictions": f"{new_folder}/test_multi_tree_data_with_predictions{CSV_SUFFIX}",
            "validation_single_tree_data": f"{new_folder}/validation_single_tree_data{CSV_SUFFIX}",
            "test_single_tree_data": f"{new_folder}/test_single_tree_data{CSV_SUFFIX}",
    "performance_on_test_set":f"{new_folder}/overall_performance_on_test_set{CSV_SUFFIX}",
     "time_model_path": f"{new_folder}/time.model",
     "final_comparison_path": f"{new_folder}/final_performance_comp{CSV_SUFFIX}",
            "final_comparison_path_agg": f"{new_folder}/final_performance_comp_agg{CSV_SUFFIX}",
     "log_file": f"{new_folder}/ML_log_file.log","time_vi": f"{new_folder}/time_vi{CSV_SUFFIX}",
            "time_metrics": f"{new_folder}/time_metrics{CSV_SUFFIX}",
            "time_group_metrics": f"{new_folder}/time_group_metrics{CSV_SUFFIX}",
            "error_vi": f"{new_folder}/error_vi{CSV_SUFFIX}",
            "error_metrics": f"{new_folder}/error_metrics{CSV_SUFFIX}",
            "error_group_metrics": f"{new_folder}/error_group_metrics{CSV_SUFFIX}",
            "final_error_vi": f"{new_folder}/final_error_vi{CSV_SUFFIX}",
            "ML_edited_default_data_path": f"{new_folder}/ML_edited_default_data{CSV_SUFFIX}"
            }



def generate_basic_data_dict(data_for_ML, args,subsample_train = False, subsample_train_frac = -1):
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
                                  val_pct=args.val_pct,subsample_train = subsample_train, subsample_train_frac =  subsample_train_frac)
    return data_dict



def generate_single_tree_models(data_dict, file_paths, args, sampling_frac):

    error_model= ML_model(X_train= data_dict["X_train"], groups= data_dict["train_MSAs"], y_train= data_dict["y_train_err"],n_jobs=args.n_jobs,
                          path = file_paths["error_model_path"], classifier = True, name = str(sampling_frac), large_grid= args.large_grid, do_RFE= args.do_RFE, n_cv_folds= args.n_CV_folds)
    print_model_statistics(model=error_model,train_X =data_dict["X_train"],  test_X=data_dict["X_test"],y_train=data_dict["y_train_err"],
                           y_test=data_dict["y_test_err"], is_classification=True, vi_path=file_paths["error_vi"], metrics_path = file_paths["error_metrics"], group_metrics_path=file_paths["error_group_metrics"],
                           name=f"Error classification model frac ={sampling_frac}", sampling_frac = sampling_frac,test_MSAs= data_dict["full_test_data"]["msa_path"])
    time_model = ML_model(X_train=data_dict["X_train"], groups=data_dict["train_MSAs"],
                          y_train=data_dict["y_train_time"], n_jobs=args.n_jobs, path=file_paths["time_model_path"],
                          classifier=False, name=str(sampling_frac), large_grid= args.large_grid, do_RFE= args.do_RFE,n_cv_folds= args.n_CV_folds)
    print_model_statistics(model=time_model, train_X=data_dict["X_train"], test_X=data_dict["X_test"],
                           y_train=data_dict["y_train_time"],
                           y_test=data_dict["y_test_time"], is_classification=False, vi_path=file_paths["time_vi"],metrics_path = file_paths["time_metrics"],group_metrics_path=file_paths["time_group_metrics"],
                           name=f"Time regression model frac={sampling_frac}", sampling_frac = sampling_frac,test_MSAs= data_dict["full_test_data"]["msa_path"])

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



from sklearn.cluster import KMeans

def cluster(X):
    k_means = KMeans(n_clusters=5).fit(X)
    return X.groupby(k_means.labels_)\
            .transform('mean').sum(1)\
            .rank(method='dense').sub(1)\
            .astype(int).to_frame()



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


    embedding_features = [col for col in  features_data.columns if ('mds' in col or 'lle' in col in col or 'PCA' in col or 'TSNE' in col or 'iso' in col) ]
    excluded_features = [f for f in embedding_features if 'time' in f]
    included_embedding_features = [f for f in embedding_features if f not in excluded_features]

    #pca_features =  features_data[["msa_path","starting_tree_type"]+[f"feature_PCA_{i}" for i in range(10)]]

    if os.path.exists(file_paths["edited_data"]):
        edited_data = pickle.load(open(file_paths["edited_data"],'rb'))
    else:
        transformed_MSAs = []
        # for MSA in features_data["msa_path"].unique():
        #     msa_data = features_data.loc[features_data.msa_path == MSA]
        #     n_clusters = 5
        #     kmeans = KMeans(n_clusters=n_clusters).fit(msa_data[[f"feature_PCA_{i}" for i in range(10)]])
        #     centroids = {i: kmeans.cluster_centers_[i] for i in range(n_clusters)}
        #     centroids_distances = {i: np.linalg.norm(v) for i, v in centroids.items()}
        #     msa_data["clusters"] = kmeans.labels_
        #     msa_data["feature_cluster_distances"] = msa_data["clusters"].apply(lambda x: centroids_distances[x])
        #     msa_data["feature_cluster_size"] = msa_data.groupby("clusters")['msa_path'].transform('count') / len(
        #         msa_data.index)
        #     msa_data["starting_tree_rand"] = msa_data["starting_tree_type"] == 'rand'
        #     msa_data["feature_cluster_size_random"] = msa_data.groupby("clusters")['starting_tree_rand'].transform(
        #         np.mean)
        #     msa_data["feature_mean_ll_pre_cluster"] = msa_data.groupby("clusters")[
        #         "feature_tree_optimized_ll"].transform(np.mean)
        #     transformed_MSAs.append(msa_data)
        # features_data = pd.concat(transformed_MSAs)
        features_data = features_data[[col for col in features_data.columns if col not in excluded_features ]]
        logging.info("Taking absolute values of each Embedding column")
        for col in [col for col in features_data.columns if 'PCA' in col or 'TSNE' in col]:
                features_data[col] = (features_data[col].abs())
        #for col in ['feature_mds_True_stress_10_only_base','feature_mds_True_stress_3_only_base','feature_mds_True_stress_30_only_base']:
        #    features_data[col+"_log"] = np.log(features_data[col])
                #features_data[col] = features_data.groupby('msa_path')[col].transform(lambda x: (x/max(x)-np.mean(x)))
        #features_data = features_data.loc[features_data.starting_tree_type == 'rand']
        logging.info(f"Number of MSAs in feature data is {len(features_data['msa_path'].unique())}")

        logging.info(f"Enriching features data in {file_paths['features_path']} and saving to {file_paths['ML_edited_features_path']}")
        edited_data = edit_raw_data_for_ML(features_data, epsilon)
        with open(file_paths["edited_data"],"wb") as EDITED_DATA:
            pickle.dump(edited_data,EDITED_DATA)
        enriched_features_data = edited_data["non_default"]
        logging.info(f"Number of positive samples: {len(enriched_features_data.loc[enriched_features_data.is_global_max==1].index)}, Number of negative samples {len(enriched_features_data.loc[enriched_features_data.is_global_max==0].index)}")
        #enriched_default_data = edited_data["default"]
        enriched_default_data_by_params = edited_data["default_by_params"]
        enriched_features_data.to_csv(file_paths["ML_edited_features_path"], sep=CSV_SEP)
        #enriched_default_data.to_csv(file_paths["ML_edited_default_data_path"], sep=CSV_SEP)

    training_fracs = args.different_training_sizes if args.test_different_training_sizes else [-1]

    for sampling_frac in training_fracs:
        logging.info(f"****Sampling frac = {sampling_frac}****")
        data_dict = generate_basic_data_dict(edited_data["non_default"], args,subsample_train = args.test_different_training_sizes, subsample_train_frac = sampling_frac)
        time_model, error_model = generate_single_tree_models(data_dict, file_paths, args,sampling_frac)

    logging.info("Using model to predict on test data")
    test_data = apply_single_tree_models_on_data(data_dict["full_test_data"], data_dict["X_test"], time_model, error_model,
                                     file_paths["test_single_tree_data"])
    logging.info("Using model to predict on validation data")
    if len(data_dict["full_validation_data"].index)>0:
        validation_data = apply_single_tree_models_on_data(data_dict["full_validation_data"], data_dict["X_val"], time_model,
                                                     error_model,
                                                     file_paths["validation_single_tree_data"])

    default_by_params_data_performance = get_default_performance(edited_data["default_by_params"], args, data_dict["full_test_data"], out_path= file_paths["default_by_params_path"])











if __name__ == "__main__":
    main()
