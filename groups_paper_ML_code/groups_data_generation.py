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
from sklearn.neighbors import KernelDensity
from feature_extraction.feature_extraction_basic import *




def get_sampled_data(n_pars, n_rand, n_sum, i, n_sample_points, msa_data, seed,possible_spr_radius,possible_spr_cutoff,default_data = False):
    if not default_data:
        random.seed(seed+1)
        spr_radius = random.choice(possible_spr_radius)
        random.seed(seed+2)
        spr_cutoff = random.choice(possible_spr_cutoff)
        logging.info(f"Chosen SPR radius {spr_radius}, Chosen SPR cutoff {spr_cutoff}")
    else:
        spr_cutoff = -1
        spr_radius = -1
    if n_pars == -1 and n_rand == -1:
        min_n_pars = max(n_sum-20,0)
        random.seed(seed+3)
        n_pars_sample = random.randint(min_n_pars, min(n_sum,20)) # Taking a max of 20 parsimony trees
        n_rand_sample = n_sum - n_pars_sample
    else:
        n_pars_sample = n_pars
        n_rand_sample = n_rand
    print(f"n_sum {n_sum} spr radius {spr_radius} spr cutoff {spr_cutoff} n_pars {n_pars_sample}")
    logging.info(f"i = {i}/{n_sample_points}")
    if not default_data:
        sampled_data_parsimony = msa_data[(msa_data["starting_tree_type"] == "pars")&(msa_data["spr_cutoff"]==spr_cutoff)&(msa_data["spr_radius"]==spr_radius)].sample(
            n=n_pars_sample,random_state=seed+3)  #
        sampled_data_random = msa_data[(msa_data["starting_tree_type"] == "rand")&(msa_data["spr_cutoff"]==spr_cutoff)&(msa_data["spr_radius"]==spr_radius)].sample(
            n=n_rand_sample,random_state=seed+4)  # random_state=seed
    else:
        sampled_data_parsimony = msa_data[
            (msa_data["starting_tree_type"] == "pars")].sample(
            n=n_pars_sample)  # random_state=seed
        sampled_data_random = msa_data[
            (msa_data["starting_tree_type"] == "rand")].sample(
            n=n_rand_sample)  # random_state=seed
    sampled_data = pd.concat([sampled_data_random, sampled_data_parsimony])

    return sampled_data, n_pars_sample, n_rand_sample



def enrich_iteration_with_extra_metrics(curr_run_dir, sampled_data, curr_iter_general_metrics):
    final_trees_RF_distance_metrics = pd.DataFrame([generate_RF_distance_matrix_statistics_final_trees(curr_run_dir,
                                                                                                       list(
                                                                                                           sampled_data[
                                                                                                               "final_tree_topology"]),
                                                                                                       best_tree=list(
                                                                                                           sampled_data[
                                                                                                               "is_top_scoring_tree"]),
                                                                                                       prefix="feature_final_trees_level_distances_RF",
                                                                                                       ll=list(
                                                                                                           sampled_data[
                                                                                                               "final_ll"]))])
    #final_tree_embedding_metrics = pd.DataFrame([generate_embedding_distance_matrix_statistics_final_trees(
    #    (sampled_data["final_tree_topology"]), best_tree=(sampled_data["is_top_scoring_tree"]),
    #    prefix="feature_final_trees_level_distances_embedd", tree_clusters=(sampled_data["tree_clusters_ind"]),
    #    True_global_trees=True_global_max_data["final_tree_topology"],
    #    True_global_tree_clusters=True_global_max_data["tree_clusters_ind"],
    #    True_global_ll_values = True_global_max_data["final_ll"],
    #    final_trees_ll=sampled_data["final_ll"])])
    try:
        log_likelihood_diff_metrics = pd.DataFrame([get_summary_statistics_dict(
            values=[np.array(sampled_data["log_likelihood_diff"])[list(sampled_data["is_top_scoring_tree"] == False)]],
            feature_name='feature_general_ll_diff')])
    except:
        log_likelihood_diff_metrics = pd.DataFrame()
    curr_iter_general_metrics = pd.concat(
        [curr_iter_general_metrics, final_trees_RF_distance_metrics,
         log_likelihood_diff_metrics], axis=1)  # final_tree_embedding_metrics,
    return curr_iter_general_metrics



# def fit_gmm(curr_iter_general_metrics, best_tree, ll_values, name):
#     gmm_1 = GaussianMixture(n_components=1, random_state=0).fit(ll_values)
#     gmm_2 = GaussianMixture(n_components=2, random_state=0).fit(ll_values)
#     gmm_3 = GaussianMixture(n_components=3, random_state=0).fit(ll_values)
#
#     curr_iter_general_metrics[f'{name}_mean_gmm_1_ll_score']= np.mean(gmm_1.score_samples(ll_values))
#     curr_iter_general_metrics[f'{name}_mean_gmm_1_ll_score_best'] = np.mean(gmm_1.score_samples(ll_values)[best_tree])
#     curr_iter_general_metrics[f'{name}_mean_gmm_2_ll_score'] = np.mean(gmm_2.score_samples(ll_values))
#     curr_iter_general_metrics[f'{name}_mean_gmm_2_ll_score_best'] = np.mean(gmm_2.score_samples(ll_values)[best_tree])
#     curr_iter_general_metrics[f'{name}_mean_gmm_3_ll_score'] = np.mean(gmm_3.score_samples(ll_values))
#     curr_iter_general_metrics[f'{name}_mean_gmm_3_ll_best'] = np.mean(gmm_3.score_samples(ll_values)[best_tree])



def single_iteration(i, curr_run_dir, ll_epsilon, n_sample_points, seed, n_pars, n_rand, n_sum_range, is_default_data, all_sampling_results, general_features, msa_data, distinct_true_best_topologies):
    print(i)
    possible_spr_radius = list(msa_data["spr_radius"].unique())
    possible_spr_cutoff = list(msa_data["spr_cutoff"].unique())
    random.seed(seed)
    print(f"seed in iteration i={seed}")
    n_sum = random.choice(n_sum_range)
    logging.info(f"N sum={n_sum}")
    sampled_data, n_pars_sample, n_rand_sample = get_sampled_data(n_pars, n_rand, n_sum, i, n_sample_points,
                                                                  msa_data, seed, default_data=is_default_data,
                                                                  possible_spr_cutoff=possible_spr_cutoff,
                                                                  possible_spr_radius=possible_spr_radius)

    N_topologies_found = len(sampled_data[sampled_data["is_global_max"] == True]['tree_clusters_ind'].unique())
    print(f"Best topologies found by the sampling trees {N_topologies_found}")

    #True_global_max_data = overall_best_msa_data.loc[
    #    ~overall_best_msa_data.tree_clusters_ind.isin(sampled_data["tree_clusters_ind"])]

    best_ll_score = sampled_data['final_ll'].max()
    sampled_data['best_sample_ll'] = best_ll_score

    sampled_data["log_likelihood_diff"] = sampled_data["best_sample_ll"] - sampled_data[
        "final_ll"]
    ll_possibly_best_topology = sampled_data.loc[sampled_data["log_likelihood_diff"] <=ll_epsilon][
        "tree_clusters_ind"].unique()
    sampled_data["is_top_scoring_tree"] = (sampled_data["tree_clusters_ind"].isin(ll_possibly_best_topology)).astype(
        'int')  # global max definition
    sampled_data["normalized_final_ll"] = sampled_data.groupby('msa_path')["final_ll"].transform(
        lambda x: ((x - x.mean()) / x.std()))
    sampled_data_top_scoring_trees = sampled_data[sampled_data["is_top_scoring_tree"] == True]

    curr_iter_general_metrics = sampled_data.groupby(
        by=["msa_path"] + general_features).agg(
        default_final_err=('delta_ll_from_overall_msa_best_topology', np.min),
        default_final_rf_distance_from_best=('rf_from_overall_msa_best_topology', np.min),
        default_status=('is_global_max', np.max),
        feature_general_pct_best=('is_top_scoring_tree', np.mean),
        feature_n_top_scoring_trees = ('is_top_scoring_tree', np.sum),
        feature_general_n_topologies=('tree_clusters_ind', pd.Series.nunique),
        feature_general_final_ll_var=('final_ll', np.var),
        feature_general_final_ll_skew=('final_ll', skew),
        feature_general_max_ll_std=('normalized_final_ll', np.max)
    ).reset_index()

    curr_iter_general_metrics["feature_pct_diff_topologies"] = curr_iter_general_metrics['feature_general_n_topologies']/n_sum
    curr_iter_general_metrics["feature_pct_diff_topologies_best_trees"] = len(ll_possibly_best_topology)/curr_iter_general_metrics["feature_n_top_scoring_trees"]
    curr_iter_general_metrics["distinct_true_topologies_found"] = N_topologies_found
    curr_iter_general_metrics["default_pct_global_max"] = 0 if len(distinct_true_best_topologies)==0 else N_topologies_found / len(distinct_true_best_topologies)
    curr_iter_general_metrics["final_topology_obtained"] = np.max(sampled_data.loc[sampled_data['log_likelihood_diff']==0]["tree_clusters_ind"])
    curr_iter_general_metrics["feature_general_n_topologies_best_final_trees"] = len(ll_possibly_best_topology)
    print(f'default_status={curr_iter_general_metrics["default_status"]}')
    print(f"Best different trees: {pd.Series.nunique(sampled_data_top_scoring_trees['tree_clusters_ind'])}")
    print(f"Overall different trees: {pd.Series.nunique(sampled_data['tree_clusters_ind'])}")
    curr_iter_general_metrics["n_pars_trees_sampled"] = n_pars_sample
    curr_iter_general_metrics["n_rand_trees_sampled"] = n_rand_sample
    curr_iter_general_metrics["n_total_trees_sampled"] = n_sum
    curr_iter_general_metrics["frac_pars_trees_sampled"] = curr_iter_general_metrics["n_pars_trees_sampled"] / n_sum
    curr_iter_general_metrics = enrich_iteration_with_extra_metrics(curr_run_dir,sampled_data, curr_iter_general_metrics)

    all_sampling_results = pd.concat(
        [all_sampling_results, curr_iter_general_metrics])  # default_results.append(general_run_metrics)
    return all_sampling_results





def MSA_pipeline(msa_path,msa_data, curr_run_dir, ll_epsilon_values, n_sample_points,seed, n_pars, n_rand, n_sum_range, all_sampling_results,all_raw_results, general_features, ready_features ):
    if ready_features:
        logging.info("Enriching MSA data, calculating features from beggining")
        msa_data, msa_type = process_all_msa_runs(curr_run_dir,msa_path, msa_data)
        msa_features = pd.DataFrame.from_dict({msa_path: get_msa_stats(msa_path, msa_type)}, orient= 'index')
        msa_data = msa_data.merge(msa_features, on = 'msa_path')
    else:
        logging.info("Using existing MSA data from ready features")
    if len(msa_data["file_name"].unique())>1: ## use raw data only for
        all_raw_results = pd.concat(
            [all_raw_results, msa_data])
    for file in msa_data["file_name"].unique():
        default_data = True if 'iqtree' in file else False
        file_data = msa_data.loc[msa_data.file_name==file]
        logging.info(f"file = {file}")
        for ll_epsilon in ll_epsilon_values:
            ll_epsilon_msa_data = file_data.copy()
            logging.info(f"Using epsilon={ll_epsilon}")
            ll_epsilon_msa_data["ll_epsilon"] = ll_epsilon
            ll_best_topologies = ll_epsilon_msa_data.loc[ll_epsilon_msa_data["delta_ll_from_overall_msa_best_topology"] <= ll_epsilon][
                "tree_clusters_ind"].unique()
            ll_epsilon_msa_data["is_global_max"] = (ll_epsilon_msa_data["tree_clusters_ind"].isin(ll_best_topologies)).astype(
                'int')  # global max definition
            overall_best_msa_data = ll_epsilon_msa_data.loc[ll_epsilon_msa_data.is_global_max == True]
            distinct_true_best_topologies = list(overall_best_msa_data["tree_clusters_ind"].unique())
            ll_epsilon_msa_data["n_distinct_true_topologies"] = len(distinct_true_best_topologies)
            print(f"Number of distinct true topologies{len(distinct_true_best_topologies)}")


            for i in range(n_sample_points):
                seed = seed+1
                all_sampling_results = single_iteration(i,curr_run_dir,ll_epsilon, n_sample_points,seed, n_pars, n_rand, n_sum_range,default_data, all_sampling_results, general_features, ll_epsilon_msa_data,distinct_true_best_topologies)
    return all_sampling_results,all_raw_results, seed






'''
     return {'au_test_sign':relevant_line[-1], 'au_test':relevant_line[-2],
                'wsh_test_sign':relevant_line[-5],'wsh_test':relevant_line[-6],
                'wkh_test_sign':relevant_line[-7],'wkh_test':relevant_line[-8],
                'sh_test_sign': relevant_line[-9], 'sh_test': relevant_line[-10],
                'kh_test_sign': relevant_line[-11], 'kh_test': relevant_line[-12]
'''



def get_all_sampling_results(curr_run_dir, data, ll_epsilon_values, n_sample_points, seed, n_pars, n_rand, n_sum_range = [10, 20], ready_features = False
                             ):

    n_sum_limits = [int(n) for n in n_sum_range.split('_')]
    n_sum_range = list(range(n_sum_limits[0], n_sum_limits[1]+1))
    logging.info(f"Sum options {n_sum_range}")
    tree_search_features = ["spr_radius","spr_cutoff"]
    #topology_test_features = ['au_test_sign','wsh_test_sign','wkh_test_sign','sh_test_sign','kh_test_sign','au_test','wsh_test','wkh_test','sh_test','kh_test']
    general_features = ["feature_msa_n_seq", "feature_msa_n_loci", "file_name", "ll_epsilon",
                             "feature_msa_pypythia_msa_difficulty",
                             "feature_msa_gap_fracs_per_seq_var", "feature_msa_entropy_mean","best_msa_ll","best_tree_ll_per_file"
                             ]
    general_features = general_features+tree_search_features
    all_sampling_results = pd.DataFrame()
    all_raw_results = pd.DataFrame()

    logging.info(f'Total MSA to run on: {len(data["msa_path"].unique())}')

    for i,msa_path in enumerate(data["msa_path"].unique()):
        logging.info(f'msa path = {msa_path}, {i}/{len(data["msa_path"].unique())}')
        # msa_features = generate_calculations_per_MSA(msa_path, curr_run_dir, n_pars_tree_sampled=150)
        msa_data = data.loc[data.msa_path == msa_path].copy().reset_index(drop=True)

        # Filter on MSA data
        logging.info(f"LL epsilon values are: {ll_epsilon_values}")
        #:
        all_sampling_results, all_raw_results, seed = MSA_pipeline(msa_path, msa_data, curr_run_dir, ll_epsilon_values, n_sample_points, seed, n_pars, n_rand, n_sum_range,
                                                          all_sampling_results,all_raw_results, general_features, ready_features= ready_features)
        #except Exception as e:
        #    logging.error(f"Could not run on MSA {msa_path}")
        #    print(e)
    return all_sampling_results,all_raw_results





def main():
    parser = group_job_parser()
    args = parser.parse_args()
    curr_run_dir = args.curr_job_folder
    create_dir_if_not_exists(curr_run_dir)
    relevant_data = pd.read_csv(args.curr_job_raw_path, sep = '\t')
    log_file_path = os.path.join(curr_run_dir,"log_file")
    level = logging.INFO if args.level=='info' else logging.DEBUG
    logging.basicConfig(filename=log_file_path, level=level)
    logging.info("Generating results file")
    ll_epsilon_values = [float(e) for e in (args.ll_epsilon).split('_')]
    results, raw_results = get_all_sampling_results(curr_run_dir, relevant_data, n_sample_points=args.n_iterations, seed=SEED, ll_epsilon_values= ll_epsilon_values, n_pars =args.n_pars_trees, n_rand = args.n_rand_trees, n_sum_range= args.n_sum_range, ready_features= args.ready_features)
    results.to_csv(args.curr_job_group_output_path, sep= '\t')
    raw_results.to_csv(args.curr_job_group_output_raw_path, sep= '\t')




if __name__ == "__main__":
    main()
