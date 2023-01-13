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

def distribute_MSAS_over_jobs(raw_data, all_jobs_results_folder,existing_msas_folder,args):
    job_dict = {}
    msa_names = list(np.unique(raw_data["msa_path"]))
    msa_splits = np.array_split(list(msa_names), min(args.n_jobs, len(msa_names)))
    for job_ind, job_msas in enumerate(msa_splits):
        logging.info(f"Submitting job {job_ind}")
        time.sleep(10)
        curr_job_folder = os.path.join(all_jobs_results_folder, "job_" + str(job_ind))
        create_or_clean_dir(curr_job_folder)
        current_raw_data_path = os.path.join(curr_job_folder, f"job_{job_ind}_raw_data{CSV_SUFFIX}")
        current_job_group_output_path = os.path.join(curr_job_folder, f"job_{job_ind}_raw_data_with_features{CSV_SUFFIX}")
        current_raw_data = raw_data[raw_data["msa_path"].isin(job_msas)]
        current_raw_data.to_csv(current_raw_data_path, sep=CSV_SEP)

        run_command = f' python {GROUPS_FEATURE_EXTRACTION_CODE} --job_ind {job_ind} --curr_job_folder {curr_job_folder} --curr_job_raw_path {current_raw_data_path} --curr_job_group_output_path {current_job_group_output_path} {generate_argument_str(args)}'

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
                              ]+ generate_argument_list(args))
        job_dict[job_ind] = {"curr_job_group_output_path": current_job_group_output_path, "job_name": job_name}

    return job_dict

def finish_all_running_jobs(job_names):
    logging.info("Deleting all jobs")
    for job_name in job_names: # remove all remaining folders
            delete_current_job_cmd = f"qstat | grep {job_name} | xargs qdel"
            execute_command_and_write_to_log(delete_current_job_cmd, print_to_log=True)


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
    relevant_data = pd.read_csv(args.file_path, sep = '\t')
    if args.filter_on_default_data:
        logging.info("Filtering on default data")
        if not LOCAL_RUN:
            relevant_data = relevant_data[relevant_data["type"] == "default"]
        else:
            relevant_data = relevant_data.loc[relevant_data.equal_to_default_config]
    relevant_data["is_global_max"] = (relevant_data["delta_ll_from_overall_msa_best_topology"] <= 0.1).astype('int')
    relevant_data = relevant_data.loc[relevant_data.feature_msa_pypythia_msa_difficulty>0.2]
    msas = relevant_data["msa_path"].unique()[:3]
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


    results["feature_mean_ll_pars_vs_rand"] = results["feature_mean_pars_ll_diff"] / results[
        "feature_mean_rand_ll_diff"]
    results["feature_var_ll_pars_vs_rand"] = results["feature_var_pars_ll_diff"] / results[
        "feature_var_rand_ll_diff"]
    train, test, val = train_test_validation_splits(results,test_pct= 0.3, val_pct=0, msa_col_name = 'msa_path')
    #X_train = train[["feature_mean_rf_final_trees","feature_msa_n_seq","feature_msa_n_loci","feature_msa_pypythia_msa_difficulty"]]
    #X_test = test[["feature_mean_rf_final_trees","feature_msa_n_seq","feature_msa_n_loci","feature_msa_pypythia_msa_difficulty"]]
    X_train = train[[col for col in train.columns if col.startswith('feature') ]]
    X_test = test[[col for col in train.columns if col.startswith('feature') ]]#+['mean_predicted_failure']
    X_val = val[[col for col in train.columns if col.startswith('feature') ]]
    y_train = train["default_status"]
    y_test = test["default_status"]
    y_val = val["default_status"]
    groups = train["msa_path"]
    model_path = os.path.join(curr_run_dir,'group_classification_model')
    vi_path=  os.path.join(curr_run_dir,'group_classification_vi.tsv')
    metrics_path = os.path.join(curr_run_dir, 'group_classification_metrics.tsv')
    group_metrics_path = os.path.join(curr_run_dir, 'group_classification_group_metrics.tsv')
    #final_csv_path = os.path.join(curr_run_dir,"performance_on_test.tsv")

    model = ML_model(X_train, groups, y_train, n_jobs = args.n_jobs, path = model_path, classifier=True, model='lightgbm', calibrate=True, name="", large_grid = args.large_grid, do_RFE = True, n_cv_folds = 3)
    #model = lightgbm.LGBMClassifier().fit(X_train, y_train)
    #model = LogisticRegressionCV(random_state=0).fit(X_train, y_train)
    #predicted_proba_test = model['best_model'].predict_proba((model['selector'].transform(X_test)))[:, 1]
    print_model_statistics(model, X_train, X_test, X_val, y_train, y_test, y_val, is_classification = True, vi_path=vi_path,
                           metrics_path=metrics_path,
                           group_metrics_path=group_metrics_path, name = "", sampling_frac = -1, test_MSAs = test["msa_path"], feature_importance=True)

if __name__ == "__main__":
    main()
