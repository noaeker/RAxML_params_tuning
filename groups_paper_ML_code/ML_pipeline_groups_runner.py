import sys

if sys.platform == "linux" or sys.platform == "linux2":
    PROJECT_ROOT_DIRECRTORY = "/groups/pupko/noaeker/RAxML_params_tuning"
else:
    PROJECT_ROOT_DIRECRTORY = "/Users/noa/Workspace/RAxML_params_tuning"
sys.path.append(PROJECT_ROOT_DIRECRTORY)

from side_code.raxml import *
from side_code.MSA_manipulation import get_local_path
from ML_utils.ML_algorithms_and_hueristics import ML_model, print_model_statistics,train_test_validation_splits
import pickle
from side_code.file_handling import create_dir_if_not_exists, create_or_clean_dir, add_csvs_content
from groups_paper_ML_code.group_side_functions import *
from groups_paper_ML_code.groups_ML_pipeline import ML_pipeline
from side_code.code_submission import generate_argument_str, submit_linux_job, generate_argument_list, submit_local_job, execute_command_and_write_to_log
import pandas as pd
import os
import numpy as np
from groups_paper_ML_code.groups_data_generation import generate_distance_matrix
import time
import timeit
from sklearn.manifold import MDS

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






def perform_MDS(distance_mat_raw, n_components = 10):
    #distance_mat_norm = generate_distance_matrix(curr_run_directory,overall_trees)/(2*n_seq-6)
    #distance_mat_raw = generate_distance_matrix(curr_run_directory, overall_trees)
    #mds_norm = MDS(random_state=0, n_components=3, metric=True, dissimilarity='precomputed').fit(distance_mat_norm)
    mds_raw = MDS(random_state=0, n_components=n_components, metric=True, dissimilarity='precomputed').fit(distance_mat_raw)
    return mds_raw.stress_





def generate_calculations_per_MSA(curr_run_dir, relevant_data,msa_res_path,n_pars_tree_sampled = 100):
    if os.path.exists(msa_res_path):
        return pickle.load(open(msa_res_path,'rb'))
    msa_res = {}
    raxml_trash_dir = os.path.join(curr_run_dir, 'raxml_trash')
    create_dir_if_not_exists(raxml_trash_dir)
    start = timeit.default_timer()
    for msa_path in relevant_data["msa_path"].unique():
        start = timeit.default_timer()
        #print(msa_path)
        #msa_n_seq = max(relevant_data.loc[relevant_data.msa_path == msa_path]["feature_msa_n_seq"])
        pars_path = generate_n_tree_topologies(n_pars_tree_sampled, get_local_path(msa_path), raxml_trash_dir,
                                               seed=1, tree_type='pars', msa_type='AA')
        with open(pars_path) as trees_path:
            newicks = trees_path.read().split("\n")
            pars = [t for t in newicks if len(t) > 0]
            distance_mat_raw = generate_distance_matrix(curr_run_dir, pars)
            mean_dist_raw = np.mean(distance_mat_raw)
            var_dist_raw = np.var(distance_mat_raw)

            MDS_raw_10 = perform_MDS(distance_mat_raw, n_components = 10)
            MDS_raw_30 = perform_MDS(distance_mat_raw, n_components=30)
            MDS_raw_50 = perform_MDS(distance_mat_raw, n_components=50)
            MDS_raw_100 = perform_MDS(distance_mat_raw, n_components=100)

            msa_res[msa_path] = {'MDS_raw_10': MDS_raw_10,'MDS_raw_30': MDS_raw_30,'MDS_raw_50': MDS_raw_50,'MDS_raw_100': MDS_raw_100, 'mean_dist_raw': mean_dist_raw,'var_dist_raw': var_dist_raw, 'pars_trees': pars}
            create_or_clean_dir(raxml_trash_dir)
            stop = timeit.default_timer()
            print('Time: ', stop - start)
    with open(msa_res_path, 'wb') as MSA_RES:
        pickle.dump(msa_res, MSA_RES)
    return msa_res


def obtain_sampling_results(results_path, previous_results_path, relevant_data, all_jobs_running_folder, existing_msas_data_path, args):
    if not os.path.exists(results_path):
        if os.path.exists(previous_results_path):
            logging.info("Using previous results path")
            prev_results = pd.read_csv(previous_results_path, sep='\t')
            existing_MSAs = prev_results["msa_path"].unique()
            logging.info(f"Number of existing MSAs is {len(existing_MSAs)}")
            relevant_data = relevant_data.loc[~relevant_data.msa_path.isin(existing_MSAs)]
            logging.info(f"Number of Remaining MSAs is {len(relevant_data['msa_path'].unique())}")
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
        results = pd.read_csv(results_path, sep='\t', index_col=False)
    return results

def edit_existing_results(results, MSA_res_df):
    results = results.merge(MSA_res_df, on="msa_path")
    results["feature_pars_dist_vs_final_dist"] = results["mean_dist_raw"]/results["feature_mean_rf_final_trees"]
    results["feature_mean_ll_pars_vs_rand"] = results["feature_mean_pars_ll_diff"] / results[
        "feature_mean_rand_ll_diff"]




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
        relevant_data = pd.read_csv(args.file_path, sep='\t', nrows=10000)
    else:
        relevant_data = pd.read_csv(args.file_path, sep = '\t')
    if args.filter_on_default_data:
        logging.info("Filtering on default data")
    relevant_data = relevant_data[relevant_data["type"] == "default"] #Filtering only on default data
    relevant_data["is_global_max"] = (relevant_data["delta_ll_from_overall_msa_best_topology"] <= 0.1).astype('int') #global max definition
    relevant_data = relevant_data.loc[relevant_data.feature_msa_pypythia_msa_difficulty>0.2]
    if LOCAL_RUN: #Subsampling MSAs for the local run only
        msas = relevant_data["msa_path"].unique()[:20]
        relevant_data = relevant_data.loc[relevant_data.msa_path.isin(msas)]
    results_path = os.path.join(curr_run_dir,'group_results.tsv')
    previous_results_path= os.path.join(curr_run_dir,'group_results_prev.tsv')
    results = obtain_sampling_results(results_path, previous_results_path, relevant_data, all_jobs_running_folder, existing_msas_data_path, args)
    msa_res_path = os.path.join(curr_run_dir, 'MSA_MDS')
    MSA_res_dict = generate_calculations_per_MSA(curr_run_dir,  results, msa_res_path)
    #results["feature_mds_pars_vs_final"] = np.log(results["msa_path"].apply(lambda x: MSA_res_dict[x]['MDS_raw'])/results["feature_mds_rf_dist_final_trees_raw"])
    logging.info(f"Number of rows in results is {len(results.index)}")
    MSA_res_df = pd.DataFrame.from_dict(MSA_res_dict, orient='index').reset_index().drop(columns=['pars_trees']).rename(
        columns={'index': 'msa_path'})
    edit_existing_results(results, MSA_res_df)
    if args.additional_validation and os.path.exists(args.additional_validation):
        additional_validation_data = pd.read_csv(args.additional_validation, sep='\t')
        edit_existing_results(additional_validation_data, MSA_res_df)
    else:
        additional_validation_data = None


    #results["feature_var_ll_pars_vs_rand"] = results["feature_var_pars_ll_diff"] / results[
    #    "feature_var_rand_ll_diff"]


    logging.info(f"Using sample fracs = {args.sample_fracs}")
    logging.info(f"include_output_tree_features = {args.include_output_tree_features}")
    sample_fracs = args.sample_fracs if not LOCAL_RUN else [1]
    if args.add_sample_fracs:
        for sample_frac in  sample_fracs:
            ML_pipeline(results, args, curr_run_dir, sample_frac, RFE=False, large_grid= False,include_output_tree_features = args.include_output_tree_features)
    if not LOCAL_RUN:
        ML_pipeline(results, args, curr_run_dir, sample_frac=1.0, RFE=True, large_grid = True, include_output_tree_features= args.include_output_tree_features)
    elif LOCAL_RUN:
        ML_pipeline(results, args, curr_run_dir, sample_frac=1.0, RFE=True, large_grid=False,
                    include_output_tree_features=args.include_output_tree_features, additional_validation_data = additional_validation_data)
    logging.info(f"Working on MSA level features")


if __name__ == "__main__":
    main()
