import sys

if sys.platform == "linux" or sys.platform == "linux2":
    PROJECT_ROOT_DIRECRTORY = "/groups/pupko/noaeker/RAxML_params_tuning"
else:
    PROJECT_ROOT_DIRECRTORY = "/Users/noa/Workspace/RAxML_params_tuning"
sys.path.append(PROJECT_ROOT_DIRECRTORY)

from side_code.SPR_moves import *
from scipy.stats import skew, kurtosis
from sklearn.cluster import AgglomerativeClustering
import networkx as nx
from scipy.stats import spearmanr
from sklearn.decomposition import KernelPCA
from side_code.raxml import *
from side_code.basic_trees_manipulation import *
from side_code.MSA_manipulation import get_alignment_data, get_msa_name
from side_code.file_handling import create_or_clean_dir, create_dir_if_not_exists
from side_code.MSA_manipulation import get_alignment_data, alignment_list_to_df, get_msa_name, \
    get_local_path
from side_code.config import *
from ML_pipeline.features_job_functions import feature_job_parser
from sklearn.manifold import MDS, Isomap, TSNE, LocallyLinearEmbedding
from side_code.config import *
from side_code.file_handling import create_dir_if_not_exists
from ML_pipeline.side_functions import get_ML_parser
from ML_pipeline.ML_pipeline_procedures import get_average_results_on_default_configurations_per_msa,edit_raw_data_for_ML
from ML_pipeline.ML_algorithms_and_hueristics import ML_model, print_model_statistics, train_test_validation_splits,variable_importance
import pandas as pd
from sklearn.cluster import DBSCAN
import lightgbm
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
import os
import pickle
import numpy as np
import argparse
from ML_pipeline.ML_algorithms_and_hueristics import train_test_validation_splits,variable_importance, model_metrics
from sklearn.metrics import matthews_corrcoef, roc_auc_score, average_precision_score, accuracy_score, precision_score, \
    recall_score, PrecisionRecallDisplay
from side_code.file_handling import create_dir_if_not_exists, create_or_clean_dir, add_csvs_content
from side_code.config import *
from ML_pipeline.group_side_functions import *
from side_code.code_submission import generate_argument_str, submit_linux_job, generate_argument_list, submit_local_job, execute_command_and_write_to_log
from ML_pipeline.features_job_functions import features_main_parser
from side_code.MSA_manipulation import get_msa_name
import pandas as pd
import os
import numpy as np
import time
from ML_pipeline.ML_pipeline_groups import perform_MDS

def distribute_MSAS_over_jobs(raw_data, all_jobs_results_folder,existing_msas_folder,args):
    job_dict = {}
    msa_names = list(np.unique(raw_data["msa_path"]))
    logging.info(f"Total number of MSAs to work on {len(msa_names)}")
    msa_splits = np.array_split(list(msa_names), min(args.n_jobs, len(msa_names)))
    logging.info(f"Total number of jobs {len(msa_splits)}")
    for job_ind, job_msas in enumerate(msa_splits):
        logging.info(f"Submitting job {job_ind}")
        time.sleep(10)
        curr_job_folder = os.path.join(all_jobs_results_folder, "job_" + str(job_ind))
        create_or_clean_dir(curr_job_folder)
        current_raw_data_path = os.path.join(curr_job_folder, f"job_{job_ind}_raw_data{CSV_SUFFIX}")
        current_job_group_output_path = os.path.join(curr_job_folder, f"job_{job_ind}_raw_data_with_features{CSV_SUFFIX}")
        current_raw_data = raw_data[raw_data["msa_path"].isin(job_msas)]
        current_raw_data.to_csv(current_raw_data_path, sep=CSV_SEP)

        run_command = f' python {GROUPS_FEATURE_EXTRACTION_CODE} --job_ind {job_ind} --curr_job_folder {curr_job_folder} --curr_job_raw_path {current_raw_data_path} --curr_job_group_output_path {current_job_group_output_path} {generate_argument_str(args, exclude=["sample_fracs"])}'

        job_name = args.jobs_prefix + str(job_ind)
        if not LOCAL_RUN:
            curr_job_log_path = os.path.join(curr_job_folder, str(job_ind) + "_tmp_log")
            submit_linux_job(job_name, curr_job_folder, curr_job_log_path, run_command, cpus=args.cpus_per_job,
                             job_ind=job_ind,
                             queue=args.queue)
        else:
            submit_local_job(GROUPS_FEATURE_EXTRACTION_CODE,
                             ["--job_ind", str(job_ind), "--curr_job_folder", curr_job_folder, "--curr_job_raw_path",
                              current_raw_data_path,
                              "--curr_job_group_output_path", current_job_group_output_path
                              ]+ generate_argument_list(args, exclude=['sample_fracs']))
        job_dict[job_ind] = {"curr_job_group_output_path": current_job_group_output_path, "job_name": job_name}

    return job_dict

def finish_all_running_jobs(job_names):
    logging.info("Deleting all jobs")
    for job_name in job_names: # remove all remaining folders
            delete_current_job_cmd = f"qstat | grep {job_name} | xargs qdel"
            execute_command_and_write_to_log(delete_current_job_cmd, print_to_log=True)


def generate_calculations_per_MSA(curr_run_dir, relevant_data,msa_res_path):
    if os.path.exists(msa_res_path):
        return pickle.load(open(msa_res_path,'rb'))
    msa_res = {}
    raxml_trash_dir = os.path.join(curr_run_dir, 'raxml_trash')
    create_dir_if_not_exists(raxml_trash_dir)
    for msa_path in relevant_data["msa_path"].unique():
        print(msa_path)
        msa_n_seq = max(relevant_data.loc[relevant_data.msa_path == msa_path]["feature_msa_n_seq"])
        pars_path = generate_n_tree_topologies(300, get_local_path(msa_path), raxml_trash_dir,
                                               seed=1, tree_type='pars', msa_type='AA')
        with open(pars_path) as trees_path:
            newicks = trees_path.read().split("\n")
            pars = [t for t in newicks if len(t) > 0]
            MDS_res_10 = perform_MDS(curr_run_dir, pars, msa_n_seq, n_components=10)
            MDS_raw_10 =MDS_res_10.iloc[0]
            mean_dist_raw = MDS_res_10.iloc[1]
            MDS_res_30 = perform_MDS(curr_run_dir, pars, msa_n_seq, n_components=30)
            MDS_raw_30 = MDS_res_30.iloc[0]
            MDS_res_50 = perform_MDS(curr_run_dir, pars, msa_n_seq, n_components=50)
            MDS_raw_50 = MDS_res_50.iloc[0]
            MDS_res_100 = perform_MDS(curr_run_dir, pars, msa_n_seq, n_components=100)
            MDS_raw_100 = MDS_res_100.iloc[0]
            msa_res[msa_path] = {'MDS_raw_10': MDS_raw_10,'MDS_raw_30': MDS_raw_30,'MDS_raw_50': MDS_raw_50,'MDS_raw_100': MDS_raw_100, 'mean_dist_raw': mean_dist_raw, 'pars_trees': pars}
            create_or_clean_dir(raxml_trash_dir)
    with open(msa_res_path, 'wb') as MSA_RES:
        pickle.dump(msa_res, MSA_RES)
    return msa_res


def ML_pipeline(results, args,curr_run_dir, sample_frac,RFE, large_grid,include_output_tree_features):
    name = f'M_frac_{sample_frac}_RFE_{RFE}_large_grid_{large_grid}_out_features_{include_output_tree_features}'
    train, test, val = train_test_validation_splits(results, test_pct=0.3, val_pct=0, msa_col_name='msa_path',subsample_train=True, subsample_train_frac= sample_frac)

    known_output_features = ["frac_pars_trees_sampled","feature_msa_n_seq", "feature_msa_n_loci", "feature_msa_pypythia_msa_difficulty",
                             "feature_msa_gap_fracs_per_seq_var", "feature_msa_entropy_mean",
                             ]

    MDS_features=  [col for col in results.columns if "mds" in col or "MDS" in col]+["mean_dist_raw","feature_pars_dist"]

    final_trees_features = ["feature_pct_best","feature_max_rf_final_trees",
                       "feature_min_rf_final_trees","feature_25_rf_final_trees","feature_75_rf_final_trees", "feature_mean_rf_final_trees",
                       "feature_var_rf_final_trees","feature_max_ll_std","feature_final_ll_var","feature_final_ll_skew","feature_final_ll_kutosis"
                       ] #"feature_mds_rf_dist_final_trees_raw",
    ll_features_to_starting_trees = ["feature_mean_rand_global_max","feature_mean_pars_global_max","feature_mean_rand_ll_diff","feature_mean_pars_ll_diff","feature_var_pars_ll_diff","feature_var_rand_ll_diff"]
    rf_features_to_starting_trees = ["feature_min_pars_vs_final_rf_diff","feature_max_pars_vs_final_rf_diff","feature_mean_pars_rf_diff"]

    combining_features = ["feature_pars_dist_vs_final_dist","feature_mean_ll_pars_vs_rand"]

    full_features = known_output_features+MDS_features+final_trees_features+ll_features_to_starting_trees+rf_features_to_starting_trees+combining_features
    MSA_level_features = known_output_features+MDS_features
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
    model_path = os.path.join(curr_run_dir, 'group_classification_model')
    vi_path = os.path.join(curr_run_dir, f'group_classification_vi_{name}.tsv')
    metrics_path = os.path.join(curr_run_dir, 'group_classification_metrics.tsv')
    group_metrics_path = os.path.join(curr_run_dir, 'group_classification_group_metrics.tsv')

    model = ML_model(X_train, groups, y_train, n_jobs=args.cpus_per_main_job, path=model_path, classifier=True, model='lightgbm',
                     calibrate=True, name=name, large_grid=large_grid, do_RFE=RFE, n_cv_folds=args.n_cv_folds)

    print_model_statistics(model, X_train, X_test, X_val, y_train, y_test, y_val, is_classification=True,
                           vi_path=vi_path,
                           metrics_path=metrics_path,
                           group_metrics_path=group_metrics_path, name=name, sampling_frac=sample_frac, test_MSAs=test["msa_path"],
                           feature_importance=True)
    if large_grid and RFE:
        test["uncalibrated_prob"] = model['best_model'].predict_proba((model['selector']).transform(X_test))[:, 1]
        test["calibrated_prob"] = model['calibrated_model'].predict_proba((model['selector']).transform(X_test))[:, 1]
        final_csv_path = os.path.join(curr_run_dir, f"{name}_final_performance_on_test.tsv")
        test.to_csv(final_csv_path, sep='\t')


def main():

    parser = group_main_parser()
    args = parser.parse_args()
    curr_run_dir = os.path.join(args.curr_working_dir, args.name)
    create_dir_if_not_exists(curr_run_dir)
    log_file_path = os.path.join(curr_run_dir,"log_file")
    level = logging.INFO if args.level=='info' else logging.DEBUG
    logging.basicConfig(filename=log_file_path, level=level)
    all_jobs_running_folder = os.path.join(curr_run_dir,'jobs')
    create_dir_if_not_exists(all_jobs_running_folder)
    existing_msas_data_path = os.path.join(curr_run_dir,'MSAs')
    create_dir_if_not_exists(existing_msas_data_path)
    logging.info(f"Reading all data from {args.file_path}")
    if LOCAL_RUN:
        relevant_data = pd.read_csv(args.file_path, sep='\t',nrows=1)
    else:
        relevant_data = pd.read_csv(args.file_path, sep = '\t')
    if args.filter_on_default_data:
        logging.info("Filtering on default data")
    relevant_data = relevant_data[relevant_data["type"] == "default"]
    relevant_data["is_global_max"] = (relevant_data["delta_ll_from_overall_msa_best_topology"] <= 0.1).astype('int')
    relevant_data = relevant_data.loc[relevant_data.feature_msa_pypythia_msa_difficulty>0.2]
    if LOCAL_RUN:
        msas = relevant_data["msa_path"].unique()[:20]
        relevant_data = relevant_data.loc[relevant_data.msa_path.isin(msas)]
    results_path = os.path.join(curr_run_dir,'group_results.tsv')
    if not os.path.exists(results_path):
        logging.info("Generating results file")
        jobs_dict = distribute_MSAS_over_jobs(relevant_data, all_jobs_running_folder, existing_msas_data_path, args)
        prev_number_of_jobs_done = 0
        existing_csv_paths = []
        while len(existing_csv_paths) < len(jobs_dict):
            existing_csv_paths = [jobs_dict[job_ind]["curr_job_group_output_path"] for job_ind in jobs_dict if
                                  os.path.exists(jobs_dict[job_ind]["curr_job_group_output_path"])]
            if len(existing_csv_paths) > prev_number_of_jobs_done:
                prev_number_of_jobs_done = len(existing_csv_paths)
                logging.info(f"total jobs done = {len(existing_csv_paths)}")
                # add_csvs_content(existing_csv_paths, features_out_path)
        all_csv_paths = [jobs_dict[job_ind]["curr_job_group_output_path"] for job_ind in jobs_dict]
        logging.info(f"done with all jobs! writing to csv in {results_path}")
        time.sleep(60)
        if not LOCAL_RUN:
            job_names = [jobs_dict[job_ind]["job_name"] for job_ind in jobs_dict]
            finish_all_running_jobs(job_names)
        results = add_csvs_content(all_csv_paths, results_path)
    else:
        logging.info("Reading existing results file")
        results = pd.read_csv(results_path, sep = '\t', index_col= False)

    msa_res_path = os.path.join(curr_run_dir, 'MSA_MDS')
    MSA_res_dict = generate_calculations_per_MSA(curr_run_dir,  results, msa_res_path)
    #results["feature_mds_pars_vs_final"] = np.log(results["msa_path"].apply(lambda x: MSA_res_dict[x]['MDS_raw'])/results["feature_mds_rf_dist_final_trees_raw"])
    logging.info(f"Number of rows in results is {len(results.index)}")
    MSA_res_df = pd.DataFrame.from_dict(MSA_res_dict, orient='index').reset_index().drop(columns=['pars_trees']).rename(
        columns={'index': 'msa_path'})
    results = results.merge(MSA_res_df, on = "msa_path")
    results["feature_pars_dist_vs_final_dist"] = results["msa_path"].apply(lambda x: MSA_res_dict[x]['mean_dist_raw'])/results["feature_mean_rf_dist_final_trees_raw"]
    results["feature_mean_ll_pars_vs_rand"] = results["feature_mean_pars_ll_diff"] / results[
        "feature_mean_rand_ll_diff"]
    #results["feature_var_ll_pars_vs_rand"] = results["feature_var_pars_ll_diff"] / results[
    #    "feature_var_rand_ll_diff"]


    logging.info(f"Using sample fracs = {args.sample_fracs}")
    logging.info(f"Working on entire set of features")
    sample_fracs = args.sample_fracs if not LOCAL_RUN else [1]
    for sample_frac in  sample_fracs:
        ML_pipeline(results, args, curr_run_dir, sample_frac, RFE=False, large_grid= False,include_output_tree_features = True)
    if not LOCAL_RUN:
        ML_pipeline(results, args, curr_run_dir, sample_frac=1.0, RFE=True, large_grid = True, include_output_tree_features= True)
    logging.info(f"Working on MSA level features")
    for sample_frac in  sample_fracs:
        ML_pipeline(results, args, curr_run_dir, sample_frac, RFE=False, large_grid= False,include_output_tree_features = False)
    if not LOCAL_RUN:
        ML_pipeline(results, args, curr_run_dir, sample_frac=1.0, RFE=True, large_grid = True, include_output_tree_features= False)



if __name__ == "__main__":
    main()
