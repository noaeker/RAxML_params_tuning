import sys

if sys.platform == "linux" or sys.platform == "linux2":
    PROJECT_ROOT_DIRECRTORY = "/groups/pupko/noaeker/RAxML_params_tuning"
else:
    PROJECT_ROOT_DIRECRTORY = "/Users/noa/Workspace/RAxML_params_tuning"
sys.path.append(PROJECT_ROOT_DIRECRTORY)
from groups_paper_ML_code.features_extactor_groups import *
from scipy.stats import skew, kurtosis
import random
from side_code.file_handling import create_dir_if_not_exists, create_or_clean_dir, add_csvs_content
from groups_paper_ML_code.group_side_functions import *
import pandas as pd
import os
import numpy as np



def get_sampled_data(n_pars,n_rand,n_sum,i,n_sample_points,msa_default_data,seed):
    random.seed(seed)
    if n_pars == -1 and n_rand == -1:
        n_pars_sample = random.randint(0, n_sum)
        n_rand_sample = n_sum - n_pars_sample
    else:
        n_pars_sample = n_pars
        n_rand_sample = n_rand
    logging.info(f"i = {i}/{n_sample_points}")
    sampled_data_parsimony = msa_default_data[msa_default_data["starting_tree_type"] == "pars"].sample(
        n=n_pars_sample, random_state = seed)  # random_state=seed
    sampled_data_random = msa_default_data[msa_default_data["starting_tree_type"] == "rand"].sample(
        n=n_rand_sample, random_state = seed)  # random_state=seed
    sampled_data = pd.concat([sampled_data_random, sampled_data_parsimony])

    return sampled_data, n_pars_sample, n_rand_sample


def get_average_results_on_default_configurations_per_msa(curr_run_dir,default_data, n_sample_points, seed, n_pars, n_rand, n_sum = -1
                                                          ):

    MSA_features = ["feature_msa_n_seq", "feature_msa_n_loci",
                             "feature_msa_pypythia_msa_difficulty",
                             "feature_msa_gap_fracs_per_seq_var", "feature_msa_entropy_mean",
                             ]
    default_results = pd.DataFrame()
    for msa_path in default_data["msa_path"].unique():
        logging.info(f"msa path = {msa_path}")
        msa_features,embedding_msa_models = generate_calculations_per_MSA(msa_path, curr_run_dir, n_pars_tree_sampled=150)
        msa_default_data = default_data.loc[default_data.msa_path==msa_path] # Filter on MSA data
        print(msa_path)
        for i in range(n_sample_points):
            print(i)
            seed = seed + 1
            sampled_data,n_pars_sample,n_rand_sample = get_sampled_data(n_pars,n_rand,n_sum,i,n_sample_points,msa_default_data,seed)
            sampled_data['best_sample_ll'] = sampled_data['final_ll'].max()
            sampled_data["is_best_tree"] = sampled_data["final_ll"]>=sampled_data['best_sample_ll']-0.1

            sampled_data_good_trees = sampled_data[sampled_data["is_best_tree"]==True]
            best_trees_RF_distance_metrics = pd.DataFrame([generate_RF_distance_matrix_statistics_final_trees(curr_run_dir,list(sampled_data_good_trees["final_tree_topology"]), prefix = "best_final_trees_trees_RF")])
            best_trees_RF_distance_metrics["feature_n_topologies_best_final_trees"] = pd.Series.nunique(sampled_data_good_trees["tree_clusters_ind"])

            sampled_data["log_likelihood_diff"] = sampled_data["final_ll"] - sampled_data[
                "feature_tree_optimized_ll"]
            sampled_data["start_vs_end"] = get_rf_dist_between_cols(sampled_data["starting_tree_object"],sampled_data["final_tree_topology"], curr_run_dir)
            sampled_data["normalized_final_ll"] = sampled_data.groupby('msa_path')["final_ll"].transform(lambda x: (x-x.mean()/x.std()))
            #sampled_data["final_o"] = sampled_data["final_tree_topology"].apply(lambda x:Tree(x, format=1))

            curr_iter_general_metrics = sampled_data.groupby(
                by=["msa_path"]+MSA_features).agg( default_final_err = ('delta_ll_from_overall_msa_best_topology', np.min),
                                                    default_status = ('is_global_max',np.max),
                                                   feature_pct_best=('is_best_tree', np.mean),
                                                   feature_n_topologies = ('tree_clusters_ind',pd.Series.nunique),
                                                    feature_final_ll_var = ('final_ll', np.var),
                                                    feature_final_ll_skew=('final_ll', skew),
                                                    feature_max_ll_std = ('normalized_final_ll', np.max)
                                                    ).reset_index()
            curr_iter_general_metrics["n_pars_trees_sampled"] = n_pars_sample
            curr_iter_general_metrics["n_rand_trees_sampled"] = n_rand_sample
            curr_iter_general_metrics["frac_pars_trees_sampled"] = curr_iter_general_metrics["n_pars_trees_sampled"] / n_sum



            final_trees_RF_distance_metrics = pd.DataFrame([generate_RF_distance_matrix_statistics_final_trees(curr_run_dir,list(sampled_data["final_tree_topology"]), prefix = "feature_final_trees_level_RF")])
            final_tree_embedding_metrics = pd.DataFrame([generate_embedding_distance_matrix_statistics_final_trees(list(sampled_data["final_tree_topology"]),models_dict=embedding_msa_models, prefix = "feature_final_trees_level_new_")])



            pars_run_metrics = sampled_data.loc[sampled_data.starting_tree_type=='pars'].groupby('msa_path').agg(feature_mean_pars_ll_diff = ('log_likelihood_diff', np.mean),feature_var_pars_ll_diff = ('log_likelihood_diff', np.var), feature_mean_pars_rf_diff = ('start_vs_end', np.mean), feature_var_pars_vs_final_rf_diff = ('start_vs_end', np.var),feature_min_pars_vs_final_rf_diff = ('start_vs_end', np.min),feature_max_pars_vs_final_rf_diff = ('start_vs_end', np.max),  feature_pars_ll_skew=('feature_tree_optimized_ll', skew),
                                                    feature_pars_ll_kutosis=('feature_tree_optimized_ll', kurtosis), feature_mean_pars_global_max = ('is_best_tree', np.mean)).reset_index()

            rand_run_metrics = sampled_data.loc[sampled_data.starting_tree_type == 'rand'].groupby('msa_path').agg(
                feature_mean_rand_ll_diff=('log_likelihood_diff', np.mean),
                feature_var_rand_ll_diff=('log_likelihood_diff', np.var),
                feature_mean_rand_global_max=('is_best_tree', np.mean)
                ).reset_index()
            msa_features_df = pd.DataFrame([msa_features])
            curr_iter_general_metrics = pd.concat(
                [curr_iter_general_metrics,final_trees_RF_distance_metrics,final_tree_embedding_metrics,best_trees_RF_distance_metrics,pars_run_metrics,rand_run_metrics,msa_features_df ], axis=1) # Adding all features together

            default_results = pd.concat([default_results,curr_iter_general_metrics])#default_results.append(general_run_metrics)
    return default_results





def main():
    parser = group_job_parser()
    args = parser.parse_args()
    curr_run_dir = args.curr_job_folder
    create_dir_if_not_exists(curr_run_dir)
    relevant_data = pd.read_csv(args.curr_job_raw_path, sep = '\t')
    #relevant_data = relevant_data.loc[relevant_data.feature_msa_pypythia_msa_difficulty>0.2]
    #msas = relevant_data["msa_path"].unique()[:10]
    #relevant_data = relevant_data.loc[relevant_data.msa_path.isin(msas)]
    log_file_path = os.path.join(curr_run_dir,"log_file")
    level = logging.INFO if args.level=='info' else logging.DEBUG
    logging.basicConfig(filename=log_file_path, level=level)
    logging.info("Generating results file")
    results = get_average_results_on_default_configurations_per_msa(curr_run_dir,relevant_data, n_sample_points=args.n_iterations, seed=1, n_pars =args.n_pars_trees, n_rand = args.n_rand_trees, n_sum = args.n_sum)
    results.to_csv(args.curr_job_group_output_path, sep= '\t')




if __name__ == "__main__":
    main()
