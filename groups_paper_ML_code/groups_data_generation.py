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
        random.seed(seed)
        spr_radius = random.choice(possible_spr_radius)
        random.seed(seed+1)
        spr_cutoff = random.choice(possible_spr_cutoff)
        logging.info(f"Chosen SPR radius {spr_radius}, Chosen SPR cutoff {spr_cutoff}")
    if n_pars == -1 and n_rand == -1:
        min_n_pars = max(n_sum-20,0)
        random.seed(seed+2)
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



def enrich_iteration_with_extra_metrics(curr_run_dir, sampled_data, True_global_max_data, curr_iter_general_metrics):
    final_trees_RF_distance_metrics = pd.DataFrame([generate_RF_distance_matrix_statistics_final_trees(curr_run_dir,
                                                                                                       list(
                                                                                                           sampled_data[
                                                                                                               "final_tree_topology"]),
                                                                                                       best_tree=list(
                                                                                                           sampled_data[
                                                                                                               "is_best_tree"]),
                                                                                                       prefix="feature_final_trees_level_distances_RF",
                                                                                                       ll=list(
                                                                                                           sampled_data[
                                                                                                               "final_ll"]))])
    final_tree_embedding_metrics = pd.DataFrame([generate_embedding_distance_matrix_statistics_final_trees(
        (sampled_data["final_tree_topology"]), best_tree=(sampled_data["is_best_tree"]),
        prefix="feature_final_trees_level_distances_embedd", tree_clusters=(sampled_data["tree_clusters_ind"]),
        True_global_trees=True_global_max_data["final_tree_topology"],
        True_global_tree_clusters=True_global_max_data["tree_clusters_ind"],
        True_global_ll_values = True_global_max_data["final_ll"],
        final_trees_ll=sampled_data["final_ll"])])
    try:
        log_likelihood_diff_metrics = pd.DataFrame([get_summary_statistics_dict(
            values=[np.array(sampled_data["log_likelihood_diff"])[list(sampled_data["is_best_tree"] == False)]],
            feature_name='feature_general_ll_diff')])
    except:
        log_likelihood_diff_metrics = pd.DataFrame()
    curr_iter_general_metrics = pd.concat(
        [curr_iter_general_metrics, final_trees_RF_distance_metrics, final_tree_embedding_metrics,
         log_likelihood_diff_metrics], axis=1)  # pars_run_metrics,rand_run_metrics # Adding all features together
    return curr_iter_general_metrics

def single_iteration(i,curr_run_dir, n_sample_points,seed, n_pars, n_rand, n_sum_range,default_data, possible_spr_cutoff,possible_spr_radius,all_sampling_results, general_features, msa_data,overall_best_msa_data):
    print(i)
    random.seed(seed)
    print(f"seed in iteration i={seed}")
    n_sum = random.choice(n_sum_range)
    logging.info(f"N sum={n_sum}")
    sampled_data, n_pars_sample, n_rand_sample = get_sampled_data(n_pars, n_rand, n_sum, i, n_sample_points,
                                                                  msa_data, seed, default_data=default_data,
                                                                  possible_spr_cutoff=possible_spr_cutoff,
                                                                  possible_spr_radius=possible_spr_radius)

    topologies_found = len(sampled_data[sampled_data["is_global_max"] == True]['tree_clusters_ind'].unique())
    print(f"Best topologies found by the sampling trees {topologies_found}")
    True_global_max_data = overall_best_msa_data.loc[
        ~overall_best_msa_data.tree_clusters_ind.isin(sampled_data["tree_clusters_ind"])]

    best_ll_score = sampled_data['final_ll'].max()
    sampled_data['best_sample_ll'] = best_ll_score

    sampled_data["log_likelihood_diff"] = sampled_data["best_sample_ll"] - sampled_data[
        "final_ll"]
    ll_possibly_best_topologies = sampled_data.loc[sampled_data["log_likelihood_diff"] <= 0.1][
        "tree_clusters_ind"].unique()
    sampled_data["is_best_tree"] = (sampled_data["tree_clusters_ind"].isin(ll_possibly_best_topologies)).astype(
        'int')  # global max definition
    sampled_data["is_best_tree_True"] = (
        sampled_data["tree_clusters_ind"].isin(ll_possibly_best_topologies)).astype(
        'int')  # global max definition
    sampled_data["normalized_final_ll"] = sampled_data.groupby('msa_path')["final_ll"].transform(
        lambda x: ((x - x.mean()) / x.std()))
    sampled_data_good_trees = sampled_data[sampled_data["is_best_tree"] == True]

    curr_iter_general_metrics = sampled_data.groupby(
        by=["msa_path"] + general_features).agg(
        default_final_err=('delta_ll_from_overall_msa_best_topology', np.min),
        default_final_rf_distance_from_best=('rf_from_overall_msa_best_topology', np.min),
        default_status=('is_global_max', np.max),
        feature_general_pct_best=('is_best_tree', np.mean),
        feature_general_n_topologies=('tree_clusters_ind', pd.Series.nunique),
        feature_general_final_ll_var=('final_ll', np.var),
        feature_general_final_ll_skew=('final_ll', skew),
        feature_general_max_ll_std=('normalized_final_ll', np.max)
    ).reset_index()

    ll_values = np.array(sampled_data["normalized_final_ll"])
    non_best_ll_values = ll_values[sampled_data["is_best_tree"]==False]
    best_ll_values = ll_values[sampled_data["is_best_tree"]==True]
    if len(non_best_ll_values)>0 and len(best_ll_values)>0:
        kde = KernelDensity(kernel='gaussian', bandwidth=1).fit(non_best_ll_values.reshape(-1,1))
        #best_ll_density = ll_density[best]

        ll_density = np.min(kde.score_samples(best_ll_values.reshape(-1,1)))
    else:
        ll_density = None
    curr_iter_general_metrics["feature_ll_density_mean"] = ll_density

    print(f"ll density = {ll_density}")
    curr_iter_general_metrics["feature_pct_diff_topologies"] = curr_iter_general_metrics['feature_general_n_topologies']/n_sum

    #curr_iter_general_metrics["default_pct_global_max"] = topologies_found / distinct_true_best_topologies
    #print(curr_iter_general_metrics["default_pct_global_max"])
    print(f'default_status={curr_iter_general_metrics["default_status"]}')
    curr_iter_general_metrics["feature_general_n_topologies_best_final_trees"] = pd.Series.nunique(
        sampled_data_good_trees["tree_clusters_ind"])
    print(f"Best different trees: {pd.Series.nunique(sampled_data_good_trees['tree_clusters_ind'])}")
    print(f"Overall different trees: {pd.Series.nunique(sampled_data['tree_clusters_ind'])}")
    curr_iter_general_metrics["n_pars_trees_sampled"] = n_pars_sample
    curr_iter_general_metrics["n_rand_trees_sampled"] = n_rand_sample
    curr_iter_general_metrics["n_total_trees_sampled"] = n_sum
    curr_iter_general_metrics["frac_pars_trees_sampled"] = curr_iter_general_metrics["n_pars_trees_sampled"] / n_sum
    curr_iter_general_metrics = enrich_iteration_with_extra_metrics(curr_run_dir,sampled_data, True_global_max_data, curr_iter_general_metrics)

    all_sampling_results = pd.concat(
        [all_sampling_results, curr_iter_general_metrics])  # default_results.append(general_run_metrics)
    return all_sampling_results

def MSA_pipeline(msa_path,i,data, curr_run_dir, n_sample_points,seed, n_pars, n_rand, n_sum_range,default_data, possible_spr_cutoff,possible_spr_radius,all_sampling_results, general_features, simulated, msa_type, program ):
    logging.info(f'msa path = {msa_path}, {i}/{len(data["msa_path"].unique())}')
    #msa_features = generate_calculations_per_MSA(msa_path, curr_run_dir, n_pars_tree_sampled=150)
    msa_data = data.loc[data.msa_path == msa_path].reset_index()  # Filter on MSA data

    logging.info("Enriching MSA data")
    msa_data = process_all_msa_runs(curr_run_dir,msa_path, msa_data, cpus_per_job = 1, msa_type=msa_type, program=program,
                         perform_topology_tests=False, simulated= simulated)
    msa_features = pd.DataFrame.from_dict({msa_path: get_msa_stats(msa_path, msa_type)}, orient= 'index')
    msa_data = msa_data.merge(msa_features, on = 'msa_path')

    ll_best_topologies = msa_data.loc[msa_data["delta_ll_from_overall_msa_best_topology"] <= 0.1][
        "tree_clusters_ind"].unique()
    msa_data["is_global_max"] = (msa_data["tree_clusters_ind"].isin(ll_best_topologies)).astype(
        'int')  # global max definition
    overall_best_msa_data = msa_data.loc[msa_data.is_global_max == True]
    distinct_true_best_topologies = len(overall_best_msa_data["tree_clusters_ind"].unique())
    msa_data["distinct_true_topologies"] = distinct_true_best_topologies
    print(f"distinct true topologies{distinct_true_best_topologies}")

    #final_tree_embedding_metrics = pd.DataFrame([generate_embedding_distance_matrix_statistics_final_trees(
    #    (msa_data["final_tree_topology"]), best_tree=(msa_data["is_global_max"]),
    #    prefix="TRUE_final_trees_level_distances_embedd", tree_clusters=(msa_data["tree_clusters_ind"]),
    #    True_global_trees=None,
    #    True_global_tree_clusters=None,
    #    final_trees_ll=msa_data["final_ll"])])


    for i in range(n_sample_points):
        seed = seed+1
        all_sampling_results = single_iteration(i,curr_run_dir, n_sample_points,seed, n_pars, n_rand, n_sum_range,default_data, possible_spr_cutoff,possible_spr_radius,all_sampling_results, general_features, msa_data,overall_best_msa_data)
    return all_sampling_results, seed


def get_all_sampling_results(curr_run_dir, data, n_sample_points, seed, n_pars, n_rand, n_sum_range = [10, 20], default_data = True, msa_type = 'AA', simulated = False, program = 'RAxML'
                             ):

    n_sum_limits = [int(n) for n in n_sum_range.split('_')]
    n_sum_range = list(range(n_sum_limits[0], n_sum_limits[1]+1))
    logging.info(f"Sum options {n_sum_range}")
    tree_search_features = ["spr_radius","spr_cutoff"]
    possible_spr_radius = list(data["spr_radius"].unique())
    possible_spr_cutoff = list(data["spr_cutoff"].unique())
    general_features = ["feature_msa_n_seq", "feature_msa_n_loci",
                             "feature_msa_pypythia_msa_difficulty",
                             "feature_msa_gap_fracs_per_seq_var", "feature_msa_entropy_mean","best_msa_ll"
                             ]
    if not default_data:
        general_features = general_features+tree_search_features
    all_sampling_results = pd.DataFrame()

    logging.info(f'Total MSA to run on: {len(data["msa_path"].unique())}')


    for i,msa_path in enumerate(data["msa_path"].unique()):
        print(msa_path)
        try:
            all_sampling_results, seed = MSA_pipeline(msa_path, i, data, curr_run_dir, n_sample_points, seed, n_pars, n_rand, n_sum_range, default_data,
                         possible_spr_cutoff, possible_spr_radius, all_sampling_results, general_features, msa_type= msa_type, simulated= simulated, program = program)
        except Exception as e:
            logging.error(f"Could not run on MSA {msa_path}")
            logging.error(f"Exceprion details: {e.message}")
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
    results = get_all_sampling_results(curr_run_dir, relevant_data, n_sample_points=args.n_iterations, seed=SEED, n_pars =args.n_pars_trees, n_rand = args.n_rand_trees, default_data= args.filter_on_default_data, n_sum_range= args.n_sum_range, simulated= args.simulated, program = args.program, msa_type= args.msa_type)
    results.to_csv(args.curr_job_group_output_path, sep= '\t')




if __name__ == "__main__":
    main()
