import sys

if sys.platform == "linux" or sys.platform == "linux2":
    PROJECT_ROOT_DIRECRTORY = "/groups/pupko/noaeker/RAxML_params_tuning"
else:
    PROJECT_ROOT_DIRECRTORY = "/Users/noa/Workspace/RAxML_params_tuning"
sys.path.append(PROJECT_ROOT_DIRECRTORY)

from side_code.raxml import *
from side_code.MSA_manipulation import get_local_path
from ML_utils.ML_algorithms_and_hueristics import ML_model, print_model_statistics_pipeline,train_test_validation_splits
import pickle
from side_code.file_handling import create_dir_if_not_exists, create_or_clean_dir, add_csvs_content
from groups_paper_ML_code.group_side_functions import *
from groups_paper_ML_code.groups_ML_pipeline import ML_pipeline
from side_code.code_submission import generate_argument_str, submit_linux_job, generate_argument_list, submit_local_job, execute_command_and_write_to_log
from sklearn.manifold import MDS, Isomap, TSNE, LocallyLinearEmbedding
from sklearn.decomposition import PCA
import pandas as pd
import os
import numpy as np
from groups_paper_ML_code.groups_data_generation import generate_RF_distance_matrix
import time
from feature_extraction.feature_extraction_basic import *
import timeit
from side_code.basic_trees_manipulation import get_distances_between_leaves,generate_tree_object_from_newick
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.spatial import distance_matrix
from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering

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
        current_job_group_output_raw_path = os.path.join(curr_job_folder, f"job_{job_ind}_raw_data_with_features_raw{CSV_SUFFIX}")
        #curr_job_MSA_output_path = os.path.join(curr_job_folder, f"job_{job_ind}_MSA_data_with_features{CSV_SUFFIX}")
        current_raw_data = raw_data[raw_data["msa_path"].isin(job_msas)]
        current_raw_data.to_csv(current_raw_data_path, sep=CSV_SEP)

        run_command = f' python {GROUPS_FEATURE_EXTRACTION_CODE} --job_ind {job_ind} --curr_job_folder {curr_job_folder} --curr_job_raw_path {current_raw_data_path} --curr_job_group_output_raw_path {current_job_group_output_raw_path} --curr_job_group_output_path {current_job_group_output_path}  {generate_argument_str(args, exclude=["sample_fracs"])}'

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
                              "--curr_job_group_output_path", current_job_group_output_path, "--curr_job_group_output_raw_path",current_job_group_output_raw_path,
                              ]+ generate_argument_list(args, exclude=['sample_fracs']))
        job_dict[job_ind] = {"curr_job_group_output_path": current_job_group_output_path,"curr_job_group_output_raw_path": current_job_group_output_raw_path, "job_name": job_name}

    return job_dict

def finish_all_running_jobs(job_names):
    logging.info("Deleting all jobs")
    for job_name in job_names: # remove all remaining folders
            delete_current_job_cmd = f"qstat | grep {job_name} | xargs qdel"
            execute_command_and_write_to_log(delete_current_job_cmd, print_to_log=True)




def obtain_sampling_results(results_path, raw_results_path,previous_results_path, relevant_data, all_jobs_running_folder, existing_msas_data_path, args):
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
        all_csvs_raw_paths = [jobs_dict[job_ind]["curr_job_group_output_raw_path"] for job_ind in jobs_dict]
        logging.info(f"done with all jobs! writing to csv in {results_path}")
        time.sleep(60)
        if not LOCAL_RUN:
            job_names = [jobs_dict[job_ind]["job_name"] for job_ind in jobs_dict]
            finish_all_running_jobs(job_names)
        raw_results = add_csvs_content(all_csvs_raw_paths, raw_results_path)
        results = add_csvs_content(all_csv_paths, results_path)
    else:
        logging.info("Reading existing results file")
        results = pd.read_csv(results_path, sep='\t', index_col=False)
        raw_results = pd.read_csv(raw_results_path,sep='\t', index_col=False)
    return results,raw_results



def filter_full_data(full_data, only_validation, n_validation):
    validation_data_bool = (
            full_data["file_name"].str.contains("ps_new_msa") | full_data["file_name"].str.contains("new_msa_ds") |
            full_data["file_name"].str.contains("iqtree"))

    zou_val_data = full_data.loc[validation_data_bool].loc[~full_data["file_name"].str.contains('large')]
    zou_val_data['file_type'] = zou_val_data['file_name'].apply(
        lambda x: 'DNA' if 'new_msa_ds' in x or 'iqtree_d' in x else 'AA')
    count_per_msa = zou_val_data.groupby("msa_path")["file_name"].nunique().reset_index()
    valid_msas = count_per_msa.loc[count_per_msa.file_name == 2]["msa_path"]
    valid_msas_and_program = zou_val_data.loc[zou_val_data.msa_path.isin(valid_msas)][
        ["msa_path", "file_type"]].sort_values("msa_path").drop_duplicates()
    n_valid_msas = np.min(valid_msas_and_program.groupby('file_type')['msa_path'].nunique())
    n_samples = min(n_validation, n_valid_msas)
    logging.info(f"Subsampling {n_samples} validation MSAs from validation")
    chosen_MSAs = valid_msas_and_program.groupby('file_type').sample(n=min(n_samples, n_valid_msas))[
            "msa_path"]  # sampling 200 from each type
    validation_data = full_data.loc[
            full_data["msa_path"].isin(chosen_MSAs) | full_data["file_name"].str.contains('large')]
    if only_validation:
        return validation_data
    else:
        full_data = pd.concat([full_data.loc[~validation_data_bool], validation_data])
        return full_data


def get_msa_type(file_name):
    if 'ds_new' in file_name or 'd_iqtree' in file_name:
        msa_type = 'DNA'
    else:
        msa_type = 'AA'
    return msa_type


def get_msa_program(file_name):
    if 'iqtree' in file_name:
        program = 'IQTREE'
    else:
        program = 'RAxML'
    return program


import sklearn
def main():
    print('The scikit-learn version is {}.'.format(sklearn.__version__))


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
    logging.info(f"Reading all data from {args.raw_data_folder}")


    relevant_data =  unify_raw_data_csvs(args.raw_data_folder).reset_index()
    relevant_data = filter_full_data(relevant_data, only_validation = args.only_validation, n_validation= args.n_validation)

    relevant_data["msa_type"] = relevant_data["file_name"].apply(lambda s: get_msa_type(s) )
    relevant_data["program"] = relevant_data["file_name"].apply(lambda s: get_msa_program(s) )


    results_path = os.path.join(curr_run_dir,f'group_results.tsv')
    raw_results_path = os.path.join(curr_run_dir,f'group_results_raw.tsv')
    previous_results_path= os.path.join(curr_run_dir,f'group_results_prev.tsv')
    results, raw_results = obtain_sampling_results(results_path, raw_results_path, previous_results_path, relevant_data, all_jobs_running_folder, existing_msas_data_path, args)
    results = results.sample(frac=1)
    logging.info(f"Number of rows in results is {len(results.index)}")
    logging.info(f"Using sample fracs = {args.sample_fracs}")
    logging.info(f"include_output_tree_features = {args.include_output_tree_features}")
    sample_fracs = args.sample_fracs if not LOCAL_RUN else [1]
    if args.only_generate_data:
        logging.info("Done generating data")
        return
    ll_epsilon = args.ML_epsilon#results["ll_epsilon"].unique():
    logging.info(f"#####Results for epsilon {ll_epsilon}#####\n#\n#\n#\n#\n#\n")
    ll_epsilon_results = results.loc[results.ll_epsilon==ll_epsilon].copy()
    if args.add_sample_fracs:
        for sample_frac in  sample_fracs:
            ML_pipeline(ll_epsilon_results, args, curr_run_dir, sample_frac, RFE=False, large_grid= False,include_output_tree_features = args.include_output_tree_features, ll_epsilon = ll_epsilon)
    if (not LOCAL_RUN) and args.model!='sgd' :
        ML_pipeline(ll_epsilon_results, args, curr_run_dir, sample_frac=1.0, RFE=True, large_grid = True, include_output_tree_features= args.include_output_tree_features,ll_epsilon = ll_epsilon)
    else:
        ML_pipeline(ll_epsilon_results, args, curr_run_dir, sample_frac=1.0, RFE=False, large_grid=False,
                    include_output_tree_features=args.include_output_tree_features,ll_epsilon = ll_epsilon)
    logging.info(f"Working on MSA level features")


if __name__ == "__main__":
    main()
