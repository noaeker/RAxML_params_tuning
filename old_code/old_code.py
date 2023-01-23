def get_tree_pairs_ML_ready_data(full_data, data_feature_names):
    train_data, test_data, validation_data = train_test_validation_splits(
        full_data, test_pct=0.5, val_pct=0,msa_col_name = "msa_path")
    X_train = train_data[data_feature_names]
    y_train_err = train_data["is_better"]
    X_test = test_data[data_feature_names]
    y_test_err = test_data["is_better"]
    return {"X_train": X_train, "y_train_err": y_train_err,"X_test": X_test,
            "y_test_err": y_test_err,
            "full_test_data": test_data}

def tree_pairs_analysis(curr_run_directory,enriched_features_data,file_paths):
    default_data = enriched_features_data[enriched_features_data['type'] == 'default']
    if not os.path.exists(file_paths["tree_pairs_path"]):
        tree_pairs_data = generate_default_tree_comparisons(curr_run_directory,default_data,file_paths["tree_pairs_path"])
        tree_pairs_data.to_csv(file_paths["tree_pairs_path"], sep = CSV_SEP)
    else:
        tree_pairs_data = pd.read_csv(file_paths["tree_pairs_path"], sep=CSV_SEP)
    tree_pairs_data["is_better"] = tree_pairs_data["delta_ll_from_overall_msa_best_topology"] - tree_pairs_data[
        "delta_ll_from_overall_msa_best_topology_other"] > 0.1
    tree_pairs_data["starting_tree_type"] = tree_pairs_data["starting_tree_type"].apply(lambda x: 0 if x=='rand' else 1)
    tree_pairs_data["starting_tree_type_other"] = tree_pairs_data["starting_tree_type_other"].apply(
        lambda x: 0 if x == 'rand' else 1)
    tree_pairs_data["feature_msa_type"] = tree_pairs_data["feature_msa_type"].apply(
        lambda x: 0 if x == 'AA' else 1)
    tree_pairs_data.drop(['Unnamed: 0'],axis=1,inplace = True)
    tree_pairs_features = [col for col in tree_pairs_data.columns if col not in ["starting_tree_ind","starting_tree_ind_other","delta_ll_from_overall_msa_best_topology","delta_ll_from_overall_msa_best_topology_other", "msa_path", "msa_path_other","starting_tree_object","starting_tree_object_other","delta_ll_from_overall_msa_best_topology_other","final_ll_other","final_tree_topology","final_tree_topology_other","LL_diff","rf_dist_final_trees","is_better"] ]
    ML_ready_data_tree_pairs = get_tree_pairs_ML_ready_data(tree_pairs_data, tree_pairs_features)
    error_model = classifier(ML_ready_data_tree_pairs["X_train"], ML_ready_data_tree_pairs["y_train_err"], n_jobs=1,
                             path=file_paths["pairs_error_model_path"],
                             use_lightgbm=True)
    err_var_impt = variable_importance(ML_ready_data_tree_pairs["X_train"], error_model)
    logging.info(f"Error RF variable importance: \n {err_var_impt}")
    predicted_success = error_model.predict(ML_ready_data_tree_pairs["X_test"])
    # predicted_failure_probabilities = error_model.predict_proba(ML_ready_data_tree_pairs["X_test"])[:, 0]
    err_test_metrics = model_metrics(ML_ready_data_tree_pairs["y_test_err"], predicted_success, is_classification=True)
    logging.info(f"Pairs Error RF metrics: \n {err_test_metrics}")
