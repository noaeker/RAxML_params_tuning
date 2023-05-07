
from ML_utils.ML_algorithms_and_hueristics import ML_model, print_model_statistics_pipeline,train_test_validation_splits, model_metrics
from groups_paper_ML_code.group_side_functions import *
import os
import numpy as np
from sklearn.mixture import GaussianMixture
from pandas.api.types import is_numeric_dtype



def write_data_to_csv(curr_run_dir, train, test, X_test, model, name, ):
    final_csv_path_train = os.path.join(curr_run_dir, f"train_data_{name}.tsv")
    train.to_csv(final_csv_path_train, sep='\t')

    test["uncalibrated_prob"] = model['best_model'].predict_proba((model['selector']).transform(X_test))[:, 1]
    test["calibrated_prob"] = model['calibrated_model'].predict_proba((model['selector']).transform(X_test))[:, 1]
    final_csv_path_test = os.path.join(curr_run_dir, f"final_performance_on_test_{name}.tsv")
    test.to_csv(final_csv_path_test, sep='\t')



def apply_on_external_validation_data(name,additional_validation_data, model, train, args, full_features):
    additional_validation_data_X = additional_validation_data[[col for col in train.columns if col in full_features]]
    additional_validation_data["calibrated_prob"] = model['calibrated_model'].predict_proba(
        (model['selector']).transform(additional_validation_data_X))[:, 1]

    y_prob_predicted = model['best_model'].predict_proba(
        (model['selector']).transform(additional_validation_data_X))[:, 1]
    y_predicted = model['best_model'].predict((model['selector']).transform(additional_validation_data_X))
    y_true = additional_validation_data["default_status"]

    logging.info(f"Wrote results to {args.additional_validation}")

    metrics = model_metrics(y_true, y_predicted, y_prob_predicted, group_metrics_path = None, sampling_frac =1,
                  is_classification=True,
                  groups_data=None)
    logging.info(f"{name} model metrics:\n{metrics}")



def get_full_and_MSA_features(results):
    tree_search_columns = ["frac_pars_trees_sampled","spr_radius","spr_cutoff","n_total_trees_sampled"
                             ]

    general_MSA_columns = ["feature_msa_n_seq", "feature_msa_n_loci",
                             "feature_msa_pypythia_msa_difficulty",
                             "feature_msa_gap_fracs_per_seq_var", "feature_msa_entropy_mean",
                             ]

    general_final_tree_metrics = [col for col in results if col.startswith('feature_general')]
    final_trees_distances_metrics = [col for col in results if col.startswith('feature_final_trees_level_distances')]
    MSA_level_distancs_metrics = [col for col in results if col.startswith('feature_MSA_level')]

    full_features = ["n_total_trees_sampled"]+general_MSA_columns +general_final_tree_metrics+final_trees_distances_metrics+MSA_level_distancs_metrics
    full_features = [col for col in full_features if 'SVR' not in col and ('mds' not in col or 'mds_20_rbf_svc_max' in col)]
    #full_features = ["n_total_trees_sampled"]+general_MSA_columns+[col for col in full_features if  ('MSA_level' not in col and 'LIN' not in col and 'gmm' not in col and 'KDE' not in col and 'non_best_score' not in col   and 'min_best_score' not in col and 'lin_svc' not in col and 'var_explained' not in col and 'total_var' not in col and 'feature_general_ll_diff' not in col and 'median' not in col and 'n_components' not in col and 'final_ll_skew' not in col and 'rbf_svc_weight2' not in col  and 'median' not in col and 'min' not in col and ('max' not in col or 'rbf' in col) and 'pct_75' not in col and 'pct_25' not in col  )] #and 'RF' not in col and 'feature_general_ll_diff' not in col
    MSA_level_features = tree_search_columns+general_MSA_columns+[col for col in MSA_level_distancs_metrics if 'MSA_level__var' not in col and 'MSA_level__PCA' not in col and 'median' not in col and 'pct_25' not in col and 'pct_75' not in col]
    return full_features,MSA_level_features






def ML_pipeline(results, args,curr_run_dir, sample_frac,RFE, large_grid,include_output_tree_features, ll_epsilon):
    name = f'M_frac_{sample_frac}_eps_{ll_epsilon}_RFE_{RFE}_large_grid_{large_grid}_out_features_{include_output_tree_features}'

    if args.model=='rf' or args.model=='sgd' or args.model=='logistic': #Removing NA values
        results['feature_final_trees_level_distances_embedd_PCA3_rbf_svc_mean_best_score'] = results['feature_final_trees_level_distances_embedd_PCA3_rbf_svc_mean_best_score'].fillna(1)
        results = results.fillna(-1)
        results.replace([np.inf, -np.inf], -1, inplace=True)


    results["file_name"] = results["file_name"].str.replace('\d+', '') #remove
    train, test, val_dict = train_test_validation_splits(results, test_pct=0.3, val_pct=0, msa_col_name='msa_path',subsample_train=True, subsample_train_frac= sample_frac)

    full_features, MSA_level_features = get_full_and_MSA_features(results)

    if include_output_tree_features:
        logging.info("Including output features in model")
        X_train = train[[col for col in train.columns if col in full_features]]
        X_test = test[[col for col in train.columns if col in full_features]]  # +['mean_predicted_failure']
        #X_val = val[[col for col in train.columns if col in full_features]]
    else:
        X_train = train[[col for col in train.columns if col in MSA_level_features]]
        X_test = test[[col for col in train.columns if col in MSA_level_features]]
        #X_val = val[[col for col in train.columns if col in MSA_level_features]]

    y_train = train["default_status"] #default_status
    y_test = test["default_status"]
    #y_val = val["default_status"]<ll_epsilon
    val_expanded_dict = {}
    for file in val_dict:
        val_expanded_dict[file] = {'size': len(val_dict[file]['msa_path'].unique())}
        val_expanded_dict[file]["X_val"] = val_dict[file][[col for col in train.columns if col in full_features]]
        val_expanded_dict[file]["y_val"] = val_dict[file]["default_status"]

    model_path = os.path.join(curr_run_dir, f'group_classification_model_eps_{ll_epsilon}')
    vi_path = os.path.join(curr_run_dir, f'group_classification_vi_large_grid_{large_grid}_eps_{ll_epsilon}.tsv')
    metrics_path = os.path.join(curr_run_dir, f'group_classification_metrics_eps_{ll_epsilon}.tsv')
    group_metrics_path = os.path.join(curr_run_dir, f'group_classification_group_metrics_{name}_eps_{ll_epsilon}.tsv')

    logging.info(f"Using model {args.model}")

    groups = train["msa_path"]
    model = ML_model(X_train, groups, y_train, n_jobs=args.cpus_per_main_job, path=model_path, classifier=True, model=args.model,
                     calibrate=True, name=name, large_grid=large_grid, do_RFE=RFE, n_cv_folds=args.n_cv_folds)


    print_model_statistics_pipeline(model, X_train, X_test, y_train, y_test, val_expanded_dict, is_classification=True,
                                    vi_path=vi_path,
                                    metrics_path=metrics_path,
                                    group_metrics_path=group_metrics_path, name=name, sampling_frac=sample_frac,
                                    feature_importance=True)

    if sample_frac==1 or sample_frac==-1:
        write_data_to_csv(curr_run_dir, train, test, X_test,model, name, )