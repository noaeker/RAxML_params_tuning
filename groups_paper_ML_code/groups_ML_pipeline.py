
from ML_utils.ML_algorithms_and_hueristics import ML_model, print_model_statistics,train_test_validation_splits
from groups_paper_ML_code.group_side_functions import *
import os
import numpy as np



def write_validation_and_test_to_csv(curr_run_dir,test,X_test,val,X_val,model,name,):
    test["uncalibrated_prob"] = model['best_model'].predict_proba((model['selector']).transform(X_test))[:, 1]
    test["calibrated_prob"] = model['calibrated_model'].predict_proba((model['selector']).transform(X_test))[:, 1]
    final_csv_path_test = os.path.join(curr_run_dir, f"final_performance_on_test_{name}.tsv")
    test.to_csv(final_csv_path_test, sep='\t')

    val["uncalibrated_prob"] = model['best_model'].predict_proba((model['selector']).transform(X_val))[:, 1]
    val["calibrated_prob"] = model['calibrated_model'].predict_proba((model['selector']).transform(X_val))[:, 1]
    final_csv_path_val = os.path.join(curr_run_dir, f"final_performance_on_val_{name}.tsv")
    val.to_csv(final_csv_path_val, sep='\t')


def apply_on_external_validation_data(additional_validation_data, model, train, args, full_features):
    additional_validation_data_X = additional_validation_data[[col for col in train.columns if col in full_features]]
    additional_validation_data["calibrated_prob"] = model['calibrated_model'].predict_proba(
        (model['selector']).transform(additional_validation_data_X))[:, 1]
    logging.info(f"Wrote results to {args.additional_validation}")
    additional_validation_data.to_csv(args.additional_validation, sep='\t')


def get_full_and_MSA_features(results):
    known_output_features = ["frac_pars_trees_sampled", "feature_msa_n_seq", "feature_msa_n_loci",
                             "feature_msa_pypythia_msa_difficulty",
                             "feature_msa_gap_fracs_per_seq_var", "feature_msa_entropy_mean",
                             ]

    MSA_embedding_features = [col for col in results.columns if col.startswith('final_trees_level_embedding')]
    final_trees_embedding_columns = [col for col in results.columns if col.startswith('MSA_level_embedding')]
    final_trees_features = ["feature_pct_best", "feature_max_rf_final_trees",
                            "feature_min_rf_final_trees", "feature_25_rf_final_trees", "feature_75_rf_final_trees",
                            "feature_mean_rf_final_trees",
                            "feature_var_rf_final_trees", "feature_max_ll_std", "feature_final_ll_var",
                            "feature_final_ll_skew", "feature_final_ll_kutosis"
                            ]  # "feature_mds_rf_dist_final_trees_raw",
    ll_features_to_starting_trees = ["feature_mean_rand_global_max", "feature_mean_pars_global_max",
                                     "feature_mean_rand_ll_diff", "feature_mean_pars_ll_diff",
                                     "feature_var_pars_ll_diff", "feature_var_rand_ll_diff"]
    rf_features_to_starting_trees = ["feature_min_pars_vs_final_rf_diff", "feature_max_pars_vs_final_rf_diff",
                                     "feature_mean_pars_rf_diff"]


    full_features = known_output_features + MSA_embedding_features + final_trees_embedding_columns+ final_trees_features + ll_features_to_starting_trees + rf_features_to_starting_trees
    MSA_level_features = known_output_features + MSA_embedding_features
    return full_features, MSA_level_features

def ML_pipeline(results, args,curr_run_dir, sample_frac,RFE, large_grid,include_output_tree_features, additional_validation_data):
    name = f'M_frac_{sample_frac}_RFE_{RFE}_large_grid_{large_grid}_out_features_{include_output_tree_features}'


    if args.model=='rf' or args.model=='sgd': #Removing NA values
        results = results.fillna(-1)
        results.replace([np.inf, -np.inf], -1, inplace=True)


    train, test, val = train_test_validation_splits(results, test_pct=0.3, val_pct=0, msa_col_name='msa_path',subsample_train=True, subsample_train_frac= sample_frac)

    full_features, MSA_level_features = get_full_and_MSA_features(results)
    if include_output_tree_features:
        logging.info("Including output features in model")
        X_train = train[[col for col in train.columns if col in full_features]]
        X_test = test[[col for col in train.columns if col in full_features]]  # +['mean_predicted_failure']
        X_val = val[[col for col in train.columns if col in full_features]]
    else:
        X_train = train[[col for col in train.columns if col in MSA_level_features]]
        X_test = test[[col for col in train.columns if col in MSA_level_features]]
        X_val = val[[col for col in train.columns if col in MSA_level_features]]

    y_train = train["default_status"]
    y_test = test["default_status"]
    y_val = val["default_status"]
    groups = train["msa_path"]
    model_path = os.path.join(curr_run_dir, f'group_classification_model')
    vi_path = os.path.join(curr_run_dir, f'group_classification_vi_large_grid_{large_grid}.tsv')
    metrics_path = os.path.join(curr_run_dir, f'group_classification_metrics.tsv')
    group_metrics_path = os.path.join(curr_run_dir, f'group_classification_group_metrics_{name}.tsv')

    logging.info(f"Using model {args.model}")


    model = ML_model(X_train, groups, y_train, n_jobs=args.cpus_per_main_job, path=model_path, classifier=True, model=args.model,
                     calibrate=True, name=name, large_grid=large_grid, do_RFE=RFE, n_cv_folds=args.n_cv_folds)

    print_model_statistics(model, X_train, X_test, X_val, y_train, y_test, y_val, is_classification=True,
                           vi_path=vi_path,
                           metrics_path=metrics_path,
                           group_metrics_path=group_metrics_path, name=name, sampling_frac=sample_frac, test_MSAs=test["msa_path"],
                           feature_importance=True)
    if additional_validation_data:
        apply_on_external_validation_data(additional_validation_data, model, train, args, full_features)


    if sample_frac==1 or sample_frac==-1:
        write_validation_and_test_to_csv(curr_run_dir,test,X_test,val,X_val,model,name,)