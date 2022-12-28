import pandas as pd

def apply_threshold(comparison_data,thresholds):
    per_t_results = []
    for threshold in thresholds:
        curr_comparison_data = comparison_data.copy()
        curr_comparison_data["threshold"] = threshold
        curr_comparison_data = curr_comparison_data.loc[curr_comparison_data.survival_probability<=threshold].sort_values(["msa_path","n_trees_rsf"]).groupby(["msa_path"]).head(1)
        t_res = curr_comparison_data.groupby(['threshold','model_name']).agg(n_MSAs = ('msa_path', pd.Series.nunique), success_prob =('status', np.mean), err = ('delta_ll', np.mean),default_success_prob = ("mean_default_status", np.mean), default_err =('mean_default_err', np.mean), running_time = ('total_actual_time', np.mean)).reset_index()
        per_t_results.append(t_res)
    return pd.concat(per_t_results)


def evaluate_model_effectiventss(model, test_full, X_Cols, agg_default, model_name, thresholds = (0.2, 0.15, 0.1, 0.05, 0.04, 0.03, 0.02, 0.01, 0.001)):
    results_per_MSA = test_full[["msa_path"]+X_Cols].drop_duplicates()
    model_predictions = model.predict_survival_function(results_per_MSA[X_Cols]) # return_array = True
    print(f"Total {len(model_predictions)} MSAs in prediction")
    total_df = pd.DataFrame()
    for fn,(i,msa_data) in zip(model_predictions,results_per_MSA.iterrows()):
        result = pd.DataFrame({'n_trees_rsf':fn.x, 'survival_probability': fn(fn.x) })
        result["msa_path"] = msa_data["msa_path"]
        #result["starting_tree_type_bool"] = msa_data["starting_tree_type_bool"]
        total_df = total_df.append(result)
        #print(result)
        #plt.step(fn.x, fn(fn.x), where="post")
    comparison = total_df.merge(test_full,left_on=['msa_path','n_trees_rsf'], right_on=['msa_path','n_trees_used'], how = 'left')
    #plt.ylim(0, 1)g
    #plt.show()
    comparison_data=comparison[['msa_path',"n_trees_rsf","survival_probability","status","total_actual_time","delta_ll","starting_tree_type_bool"]].sort_values('msa_path')
    comparison_data["model_name"] = model_name
    comparison_data = comparison_data.merge(agg_default, on = "msa_path")
    summarized_comparison = apply_threshold(comparison_data,thresholds)
    return summarized_comparison,comparison_data
# def survival_pipeline(train, test, test_full_options, X_cols, agg_default):
#     X_train, Y_train, X_test, Y_test = get_train_test_X_y_survival(train, test,
#                                                                    X_cols=X_cols)
#
#     va_times = list(range(1, 20))
#     # gb = GB_survival_pipeline(X_train, Y_train, X_test, Y_test, va_times)
#     cph, cph_auc, cph_mean_auc = COX_survival_pipeline(X_train, Y_train, X_test, Y_test, va_times)
#
#     rsf, rsf_auc, rsf_mean_auc = RSF_survival_pipeline(X_train, Y_train, X_test, Y_test, va_times)
#
#     #plt.plot(va_times, cph_auc, "o-", label="CoxPH (mean AUC = {:.3f})".format(cph_mean_auc))
#     #plt.plot(va_times, rsf_auc, "o-", label="RSF (mean AUC = {:.3f})".format(rsf_mean_auc))
#     #plt.xlabel("Number of starting trees")
#     #plt.ylabel("time-dependent AUC")
#     #plt.legend(loc="lower center")
#     #plt.grid(True)
#     #plt.show()


    #cph_df_summarized, cph_df = evaluate_model_effectiventss(cph, test_full_options, X_cols, agg_default=agg_default, model_name="cph")
    #rsf_df_summarized, rsh_df = evaluate_model_effectiventss(rsf, test_full_options, X_cols, agg_default=agg_default, model_name="rph")
    #summarized_results = pd.concat([cph_df_summarized, rsf_df_summarized])
    #detailed_results = pd.concat([cph_df, rsh_df])
    #return summarized_results,    detailed_results
    # gb_df = evaluate_model_effectiventss(gb, test_edited_df_full,X_cols, agg_default=agg_default)

