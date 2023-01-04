from ML_pipeline.survival_analysis_functions import *
from lifelines import LogNormalAFTFitter,LogLogisticAFTFitter,WeibullAFTFitter,GeneralizedGammaRegressionFitter
from ML_pipeline.ML_algorithms_and_hueristics import train_test_validation_splits
from lifelines.utils import k_fold_cross_validation
import pandas as pd
import seaborn as sns
import numpy as np
import os
from lifelines import WeibullAFTFitter
import lifelines
from lifelines import CoxPHFitter
from lifelines.datasets import load_rossi
from lifelines.calibration import survival_probability_calibration

def pct_25(values):
    return np.percentile(values, 25)


def pct_75(values):
    return np.percentile(values, 75)


def test_precentiles(test_regression_dataset, full_test, model, default):
    all_results= []
    for thredhold in [0.01,0.02,0.03,0.04,0.05,0.1,0.15,0.2, 0.3]:
        predicted_p =model.predict_percentile(test_regression_dataset, ancillary=True, p=thredhold)
        df = full_test.copy()
        df['predicted_p'] = predicted_p
        df['predicted_p_restricted'] = df['predicted_p'].apply(lambda x: min(20,x))
        df['larger_than_20'] = df['predicted_p']>20
        df["success"] = (df.n_trees_used<=df.predicted_p)
        df["success_restricted"] = df.n_trees_used<=df.predicted_p_restricted
        df = df.merge(default, on = "msa_path", how = 'inner')
        per_msa_success = df.groupby(['msa_path','mean_default_status']).agg(max_status = ('status',max),max_success= ('success',max), total_time = ('predicted_p', np.sum), larger_than_20 = ('larger_than_20',max)).reset_index().groupby(['max_status','larger_than_20']).agg(n_msas  = ('msa_path',pd.Series.nunique), mean_success = ('max_success', np.mean), mean_default_success = ('mean_default_status', np.mean), median_time = ('total_time', np.median), min_time = ('total_time', np.min), max_time = ('total_time', np.max) ).reset_index()
        per_msa_success['threshold'] = thredhold
        #grouped_results = df.groupby(['status','larger_than_20','starting_tree_type_bool']).agg(mean_predicted_p = ('predicted_p', np.mean),median_predicted_p = ('predicted_p', np.median),p25_predicted_p = ('predicted_p', pct_25),p75_predicted_p = ('predicted_p', pct_75), mean_success= ('success', np.mean), n_MSAs = ('msa_path',pd.Series.nunique)).reset_index()
        #grouped_results["threshold"] = thredhold
        all_results.append(per_msa_success)
    final_df = pd.concat(all_results)
    return final_df



def test_parametric_approach(model, regression_dataset,test_regression_dataset, full_test, ancillary,name, default):
    aft_model = model.fit(regression_dataset, "n_trees_used", "status", ancillary= ancillary,formula="mean_success_prob+p20_success_prob+starting_tree_type_bool") #(mean_success_prob+p20_success_prob)*starting_tree_type_bool
    aft_model.print_summary()
    model.predict_hazard(test_regression_dataset.loc[test_regression_dataset.starting_tree_type_bool]).plot()
    #plt.show()
    model.predict_hazard(test_regression_dataset.loc[test_regression_dataset.starting_tree_type_bool==0]).plot()
    #plt.show()
    #model.predict_survival_function(test_regression_dataset)
    #model.predict_median(test_regression_dataset)
    final_df = test_precentiles(test_regression_dataset, full_test, model,default)
    final_df.to_csv("/Users/noa/Workspace/raxml_deep_learning_results/ready_raw_data/Pandit/ML/full_result.csv")
    aft_model.plot_partial_effects_on_outcome('starting_tree_type_bool', values=[0,1], cmap='coolwarm')
    #plt.show()
    # aft_model.plot_partial_effects_on_outcome('mean_success_prob', values=[0.1,0.3,0.5,0.8], cmap='coolwarm')
    # plt.show()
    return {'name': name,'AIC':aft_model.AIC_,  'concordance_index':aft_model.score(regression_dataset, scoring_method="concordance_index")}

def model_LL_cv(regression_dataset, model,ancillary = True):
    kf = KFold(n_splits=3)
    CV_LL_scores = []
    for train_ind, test_ind in kf.split(regression_dataset):
        cv_train = regression_dataset.iloc[train_ind]
        cv_test = regression_dataset.iloc[test_ind]
        curr_cv_model = model.fit(cv_train, "n_trees_used", "status", ancillary=ancillary)
        LL = curr_cv_model.score(cv_test)
        CV_LL_scores.append(LL)
    return np.mean(CV_LL_scores)

def survival_pipeline_new(train, test, test_full_options, default):
    X_cols = ["mean_success_prob","starting_tree_type_bool","feature_msa_pypythia_msa_difficulty","p20_success_prob","var_success_prob"]
    regression_dataset = train[X_cols+["n_trees_used","status"]]
    test_regression_dataset = test[X_cols]


    #loglogistic = test_parametric_approach(LogLogisticAFTFitter(),regression_dataset,test_regression_dataset,test,ancillary = False)
    #lognormal = test_parametric_approach(LogNormalAFTFitter(), regression_dataset, test_regression_dataset,test, ancillary = False)
    #Weibull = test_parametric_approach(WeibullAFTFitter(), regression_dataset, test_regression_dataset,test, ancillary = False)
    overall_df = pd.DataFrame()
    loglogistic_ancillary = test_parametric_approach(LogLogisticAFTFitter(),regression_dataset,test_regression_dataset,test,ancillary=True, name = 'loglogistic', default =  default)
    loglogistic_ancillary["LL_cv"] = model_LL_cv(regression_dataset,LogLogisticAFTFitter())
    #overall_df = overall_df.append(loglogistic_ancillary, ignore_index= True)
    #lognormal_ancillary = test_parametric_approach(LogNormalAFTFitter(), regression_dataset, test_regression_dataset,test,ancillary=True,name = 'lognormal', default = default)
    #lognormal_ancillary["LL_cv"] = model_LL_cv(regression_dataset, LogNormalAFTFitter())
    #overall_df = overall_df.append(lognormal_ancillary, ignore_index= True)
    #Weibull_ancillary = test_parametric_approach(WeibullAFTFitter(), regression_dataset, test_regression_dataset,test,ancillary=True,name = 'Weibull')
    #Weibull_ancillary["LL_cv"] = model_LL_cv(regression_dataset, WeibullAFTFitter())
    #overall_df = overall_df.append(Weibull_ancillary, ignore_index= True)
    print(overall_df)
    overall_df.to_csv("/Users/noa/Workspace/raxml_deep_learning_results/ready_raw_data/Pandit/ML/aft_models.tsv",sep= '\t')







def edit_data_for_survival_analysis(orig_df, iter):
    all_surv_data = pd.DataFrame()
    full_df = pd.DataFrame()
    MSA_cols = [ "feature_msa_pypythia_msa_difficulty",
                "feature_tree_parsimony_rf_values_pct_25_averaged_per_entire_MSA","feature_tree_MAD_averaged_per_entire_MSA"]
    orig_df["predicted_calibrated_success_probability"] = orig_df['predicted_calibrated_failure_probabilities'].apply(
        lambda x: 1 - x)
    orig_df["predicted_uncalibrated_success_probability"] = orig_df['predicted_uncalibrated_failure_probabilities'].apply(
        lambda x: 1 - x)
    orig_df["best_accuracy_per_starting_tree"] = orig_df.groupby(['msa_path', 'starting_tree_type', 'starting_tree_ind'])[
        'predicted_calibrated_success_probability'].transform(max)
    # for msa in df["msa_path"].unique()[:10]:
    #    msa_dat = df.loc[(df.msa_path==msa)&(df.starting_tree_ind==1)&(df.starting_tree_type=='pars')]
    #    plt.scatter(msa_dat['predicted_calibrated_success_probability'],msa_dat['predicted_time'] )
    #    plt.show()
    df = orig_df.loc[orig_df.equal_to_default_config==True]
    df = df.sort_values(['msa_path', 'starting_tree_type', 'predicted_calibrated_failure_probabilities'])
    df["cum_failure"] = df.groupby(['msa_path', 'starting_tree_type'])[
        'predicted_calibrated_failure_probabilities'].cumprod()
    df["iid_success_prob"] = df["cum_failure"].apply(lambda x: 1 - x)
    df["status"] = df.groupby(['msa_path', 'starting_tree_type'])['is_global_max'].cummax()
    df["delta_ll"] = df.groupby(['msa_path', 'starting_tree_type'])['delta_ll_from_overall_msa_best_topology'].cummin()
    df["total_actual_time"] = df.groupby(['msa_path', 'starting_tree_type'])['normalized_relative_time'].cumsum()
    df = df[MSA_cols + ["msa_path", "starting_tree_ind", "starting_tree_type", "iid_success_prob", "cum_failure",
                        "predicted_calibrated_success_probability","predicted_uncalibrated_success_probability", "predicted_calibrated_failure_probabilities",
                        "status", "total_actual_time", "delta_ll"]].sort_values(
        ["msa_path", "starting_tree_type", "predicted_calibrated_failure_probabilities"])
    df["starting_tree_type_bool"] = df["starting_tree_type"] == 'pars'
    df["max_status"] = df.groupby(["msa_path", "starting_tree_type"])["status"].transform(max)
    df["n_trees_used"] = df.groupby(["msa_path", "starting_tree_type"]).cumcount() + 1
    df["mean_success_prob"] = (df.groupby(["msa_path", "starting_tree_type"])[
        "predicted_calibrated_success_probability"].transform(np.mean))
    df["var_success_prob"] = df.groupby(["msa_path", "starting_tree_type"])[
        "predicted_calibrated_success_probability"].transform(np.var)
    #df["p20_median_success_prob"] = df.groupby(["msa_path", "starting_tree_type"])[
    #    "iid_success_prob"].transform(np.median)
    df["p20_success_prob"] = (df.groupby(["msa_path", "starting_tree_type"])[
        "iid_success_prob"].transform(np.max)**(1/20))
    #df["p20_success_prob_75"] = df.groupby(["msa_path", "starting_tree_type"])[
    #    "iid_success_prob"].transform(pct_75)
    #df["p20_success_prob_25"] = df.groupby(["msa_path", "starting_tree_type"])[
    #    "iid_success_prob"].transform(pct_25)
    successes = df.loc[df.max_status == 1]
    failures = df.loc[df.max_status == 0]
    successes_surv = successes.loc[successes.status == 1].sort_values(
        ["msa_path", "n_trees_used", "starting_tree_type"]).groupby(["msa_path", "starting_tree_type"]).head(1)
    failures_surv = failures.sort_values("n_trees_used", ascending=False).groupby(
        ["msa_path", "starting_tree_type"]).head(1)
    #failures_surv["n_trees_used"] = failures_surv["n_trees_used"]+1
    surv_data = pd.concat([successes_surv, failures_surv])
    all_surv_data = all_surv_data.append(surv_data, ignore_index= True)
    full_df = full_df.append(df, ignore_index= True)
    return surv_data,full_df




def main():
    test_data = pd.read_csv(
        "/Users/noa/Workspace/raxml_deep_learning_results/ready_raw_data/Pandit/ML/test_single_tree_data.tsv", sep='\t')
    default_data = pd.read_csv(
        "/Users/noa/Workspace/raxml_deep_learning_results/ready_raw_data/Pandit/ML/default_by_params_sampling.tsv",
        sep='\t')
    test_train_data, test_test_data, test_validation_data = train_test_validation_splits(
        test_data, test_pct=0.4, val_pct=0)

    train_out_path = 'train_edited_df.tsv'
    test_out_path = 'test_edited_df.tsv'
    if os.path.exists(train_out_path):
        train_edited_df = pd.read_csv(train_out_path, sep= '\t')
    else:
        train_edited_df, train_edited_df_full = edit_data_for_survival_analysis(test_train_data, iter = 1000)
        train_successes = len( train_edited_df.loc[ train_edited_df.status == 1]["msa_path"].unique())
        train_msas = len( train_edited_df["msa_path"].unique())
        print(f'Number of successes in train {train_successes}/{train_msas}')
    if os.path.exists(test_out_path):
        test_edited_df = pd.read_csv(test_out_path, sep='\t')
    else :
        test_edited_df, test_edited_df_full = edit_data_for_survival_analysis(test_test_data, iter = 1)
        test_successes = len(test_edited_df.loc[test_edited_df.status == 1]["msa_path"].unique())
        test_msas = len(test_edited_df["msa_path"].unique())
        print(f'Number of successes in test {test_successes}/{test_msas}')
        #KM_estimator(validation_edited_df)
    #survival_pipeline_new(train=validation_edited_df_rand, test=test_edited_df_rand, test_full_options=test_edited_df_full_rand)
    #validation_edited_df = pd.concat([validation_edited_df,test_edited_df]).reset_index()
    agg_default = default_data.groupby('msa_path').agg(mean_default_status=('default_status', np.mean),
                                                        mean_default_err=('default_final_err', np.mean))
    survival_pipeline_new(train=train_edited_df, test=test_edited_df,
                          test_full_options=test_edited_df, default =  agg_default)





    # agg_default = default_data.groupby('msa_path').agg(mean_default_status=('default_status', np.mean),
    #                                                    mean_default_err=('default_final_err', np.mean))
    # summarized_results_pars,detailed_results_pars = survival_pipeline(validation_edited_df_pars, test_edited_df_pars, test_edited_df_full_pars, X_cols, agg_default)
    # summarized_results_rand, detailed_results_rand = survival_pipeline(validation_edited_df_rand, test_edited_df_rand,
    #                                                                   test_edited_df_full_rand, X_cols, agg_default)
    #
    #
    # total_df = []
    # for pars_thresh in np.linspace(0.01,1,num=10):
    #     for rand_thresh in np.linspace(0.01,1,num=10):
    #         results_pars_filtered = detailed_results_pars.loc[detailed_results_pars.survival_probability<pars_thresh].sort_values("n_trees_rsf").groupby(['msa_path','model_name']).head(1)[['model_name','msa_path','status','total_actual_time']].rename(columns={'status':'pars_status','total_actual_time': 'total_actual_time_pars'})
    #         results_rand_filtered  = detailed_results_rand.loc[detailed_results_rand.survival_probability<rand_thresh].sort_values("n_trees_rsf").groupby(['msa_path','model_name']).head(1)[['model_name','msa_path','status','total_actual_time']].rename(columns={'status':'rand_status','total_actual_time': 'total_actual_time_rand'})
    #         final_res = results_pars_filtered.merge(results_rand_filtered, on = ['msa_path','model_name'])
    #         final_res = final_res.merge(agg_default, on = "msa_path")
    #         final_res["total_status"] = final_res[["pars_status","rand_status"]].max(axis=1)
    #         final_res["total_time"] = final_res["total_actual_time_pars"]+ final_res["total_actual_time_rand"]
    #         sum_final_res = final_res.groupby(['model_name']).agg(mean_total_status = ('total_status',np.mean),default_success_prob = ("mean_default_status", np.mean), n_MSAs = ('msa_path', pd.Series.nunique), mean_total_time = ('total_time',np.mean) )
    #         sum_final_res["pars_thresh"] = pars_thresh
    #         sum_final_res["rand_thresh"] = rand_thresh
    #         total_df.append(sum_final_res)
    # pd.concat(total_df).to_csv("/Users/noa/Workspace/raxml_deep_learning_results/ready_raw_data/Pandit/ML/final_res.csv")
    # summarized_results_pars.to_csv("/Users/noa/Workspace/raxml_deep_learning_results/ready_raw_data/Pandit/ML/summarized_pars.csv")
    # summarized_results_rand.to_csv(
    #     "/Users/noa/Workspace/raxml_deep_learning_results/ready_raw_data/Pandit/ML/summarized_rand.csv")
    #




if __name__ == "__main__":
    main()


