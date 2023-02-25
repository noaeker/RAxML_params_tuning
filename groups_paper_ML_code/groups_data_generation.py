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



def get_sampled_data(n_pars, n_rand, n_sum, i, n_sample_points, msa_data, seed,possible_spr_radius,possible_spr_cutoff,default_data = False):
    if not default_data:
        spr_radius = random.choice(possible_spr_radius)
        spr_cutoff = random.choice(possible_spr_cutoff)
        logging.info(f"Chosen SPR radius {spr_radius}, Chosen SPR cutoff {spr_cutoff}")
    random.seed(seed)
    if n_pars == -1 and n_rand == -1:
        n_pars_sample = random.randint(0, min(n_sum,20)) # Taking a max of 20 parsimony trees
        n_rand_sample = n_sum - n_pars_sample
    else:
        n_pars_sample = n_pars
        n_rand_sample = n_rand
    logging.info(f"i = {i}/{n_sample_points}")
    if not default_data:
        sampled_data_parsimony = msa_data[(msa_data["starting_tree_type"] == "pars")&(msa_data["spr_cutoff"]==spr_cutoff)&(msa_data["spr_radius"]==spr_radius)].sample(
            n=n_pars_sample, random_state = seed)  # random_state=seed
        sampled_data_random = msa_data[(msa_data["starting_tree_type"] == "rand")&(msa_data["spr_cutoff"]==spr_cutoff)&(msa_data["spr_radius"]==spr_radius)].sample(
            n=n_rand_sample, random_state = seed)  # random_state=seed
    else:
        sampled_data_parsimony = msa_data[
            (msa_data["starting_tree_type"] == "pars")].sample(
            n=n_pars_sample, random_state=seed)  # random_state=seed
        sampled_data_random = msa_data[
            (msa_data["starting_tree_type"] == "rand")].sample(
            n=n_rand_sample, random_state=seed)  # random_state=seed
    sampled_data = pd.concat([sampled_data_random, sampled_data_parsimony])

    return sampled_data, n_pars_sample, n_rand_sample




def get_average_results_on_default_configurations_per_msa(curr_run_dir, data, n_sample_points, seed, n_pars, n_rand, n_sum_range = [10, 20], default_data = True,
                                                          ):

    n_sum_limits = [int(n) for n in n_sum_range.split('_')]
    n_sum_range = list(range(n_sum_limits[0], n_sum_limits[1]+1))
    logging.info(f"Sum options {n_sum_range}")
    tree_search_features = ["spr_radius","spr_cutoff"]
    possible_spr_radius = list(data["spr_radius"].unique())
    possible_spr_cutoff = list(data["spr_cutoff"].unique())
    general_features = ["feature_msa_n_seq", "feature_msa_n_loci",
                             "feature_msa_pypythia_msa_difficulty",
                             "feature_msa_gap_fracs_per_seq_var", "feature_msa_entropy_mean",
                             ]
    if not default_data:
        general_features = general_features+tree_search_features
    all_sampling_results = pd.DataFrame()

    logging.info(f'Total MSA to run on: {len(data["msa_path"].unique())}')
    for i,msa_path in enumerate(data["msa_path"].unique()):
        logging.info(f'msa path = {msa_path}, {i}/{len(data["msa_path"].unique())}')
        msa_features = generate_calculations_per_MSA(msa_path, curr_run_dir, n_pars_tree_sampled=150)
        msa_data = data.loc[data.msa_path == msa_path] # Filter on MSA data
        for i in range(n_sample_points):
            print(i)
            n_sum = random.choice(n_sum_range)
            logging.info(f"N sum={n_sum}")
            seed = seed + 1

            sampled_data,n_pars_sample,n_rand_sample = get_sampled_data(n_pars,n_rand,n_sum,i,n_sample_points,msa_data,seed, default_data = default_data, possible_spr_cutoff= possible_spr_cutoff, possible_spr_radius = possible_spr_radius)

            best_ll_score = sampled_data['final_ll'].max()
            sampled_data['best_sample_ll'] = best_ll_score

            best_msa_tree_topology = max(sampled_data[sampled_data["final_ll"] == best_ll_score]['final_tree_topology'])
            sampled_data["rf_from_overall_msa_best_topology"] = msa_data["final_tree_topology"].apply(
                lambda x: rf_distance(curr_run_dir, x, best_msa_tree_topology,
                                      name="diff_from_final_tree"))
            sampled_data["is_best_tree"] = (sampled_data["final_ll"]>=sampled_data['best_sample_ll']-0.1)|(sampled_data["rf_from_overall_msa_best_topology"]==0)

            sampled_data_good_trees = sampled_data[sampled_data["is_best_tree"]==True]

            sampled_data["log_likelihood_diff"] = sampled_data["best_sample_ll"] - sampled_data[
                "final_ll"]
            log_likelihood_diff_metrics = pd.DataFrame([get_summary_statistics_dict(values=[diff for diff in sampled_data["log_likelihood_diff"] if diff>0.1], feature_name='feature_general_ll_diff')])
            print(log_likelihood_diff_metrics)
            #sampled_data["start_vs_end"] = get_rf_dist_between_cols(sampled_data["starting_tree_object"],sampled_data["final_tree_topology"], curr_run_dir)
            sampled_data["normalized_final_ll"] = sampled_data.groupby('msa_path')["final_ll"].transform(lambda x: ((x-x.mean())/x.std()))
            #sampled_data["final_o"] = sampled_data["final_tree_topology"].apply(lambda x:Tree(x, format=1))


            curr_iter_general_metrics = sampled_data.groupby(
                by=["msa_path"]+general_features).agg( default_final_err = ('delta_ll_from_overall_msa_best_topology', np.min),
                                                    default_status = ('is_global_max',np.max),
                                                   feature_general_pct_best=('is_best_tree', np.mean),
                                                   feature_general_n_topologies = ('tree_clusters_ind',pd.Series.nunique),
                                                    feature_general_final_ll_var = ('final_ll', np.var),
                                                    feature_general_final_ll_skew=('final_ll', skew),
                                                    feature_general_max_ll_std = ('normalized_final_ll', np.max)
                                                    ).reset_index()
            curr_iter_general_metrics.update(log_likelihood_diff_metrics)
            print(f'default_status={curr_iter_general_metrics["default_status"]}')
            curr_iter_general_metrics["feature_general_n_topologies_best_final_trees"] = pd.Series.nunique(sampled_data_good_trees["tree_clusters_ind"])
            print(f"Best different trees: {pd.Series.nunique(sampled_data_good_trees['tree_clusters_ind'])}")
            print(f"Notr Best different trees: {pd.Series.nunique(sampled_data['tree_clusters_ind'])}")
            curr_iter_general_metrics["n_pars_trees_sampled"] = n_pars_sample
            curr_iter_general_metrics["n_rand_trees_sampled"] = n_rand_sample
            curr_iter_general_metrics["n_total_trees_sampled"] = n_sum
            curr_iter_general_metrics["frac_pars_trees_sampled"] = curr_iter_general_metrics["n_pars_trees_sampled"] / n_sum

            final_trees_RF_distance_metrics = pd.DataFrame([generate_RF_distance_matrix_statistics_final_trees(curr_run_dir,list(sampled_data["final_tree_topology"]),best_tree= list(sampled_data["is_best_tree"]), prefix = "feature_final_trees_level_distances_RF",ll=list(sampled_data["final_ll"]))])
            final_tree_embedding_metrics = pd.DataFrame([generate_embedding_distance_matrix_statistics_final_trees(list(sampled_data["final_tree_topology"]),best_tree= list(sampled_data["is_best_tree"]), prefix = "feature_final_trees_level_distances_embedd",ll=list(sampled_data["final_ll"]))])
            msa_features_df = pd.DataFrame([msa_features])
            curr_iter_general_metrics = pd.concat(
                [curr_iter_general_metrics,final_trees_RF_distance_metrics,final_tree_embedding_metrics,msa_features_df,log_likelihood_diff_metrics ], axis=1) #pars_run_metrics,rand_run_metrics # Adding all features together

            all_sampling_results = pd.concat([all_sampling_results,curr_iter_general_metrics])#default_results.append(general_run_metrics)
    return all_sampling_results





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
    results = get_average_results_on_default_configurations_per_msa(curr_run_dir, relevant_data, n_sample_points=args.n_iterations, seed=1, n_pars =args.n_pars_trees, n_rand = args.n_rand_trees, default_data= args.filter_on_default_data, n_sum_range= args.n_sum_range)
    results.to_csv(args.curr_job_group_output_path, sep= '\t')




if __name__ == "__main__":
    main()
