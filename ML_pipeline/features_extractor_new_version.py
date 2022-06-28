import sys

if sys.platform == "linux" or sys.platform == "linux2":
    PROJECT_ROOT_DIRECRTORY = "/groups/pupko/noaeker/RAxML_params_tuning"
else:
    PROJECT_ROOT_DIRECRTORY = "/Users/noa/Workspace/RAxML_params_tuning"
sys.path.append(PROJECT_ROOT_DIRECRTORY)

from side_code.SPR_moves import *
from side_code.raxml import *
from side_code.basic_trees_manipulation import *
from side_code.MSA_manipulation import get_alignment_data, get_msa_name
from side_code.file_handling import create_or_clean_dir, create_dir_if_not_exists
from side_code.MSA_manipulation import get_alignment_data, alignment_list_to_df, get_msa_type, get_msa_name
from side_code.config import *
from shutil import rmtree
import pandas as pd
import pickle
import argparse
from pypythia.predictor import DifficultyPredictor
from pypythia.prediction import get_all_features
from pypythia.raxmlng import RAxMLNG
from pypythia.msa import MSA


def pypythia(msa_path):
    predictor = DifficultyPredictor(open(f"{PROJECT_ROOT_DIRECRTORY}/pypythia/predictor.pckl", "rb"))
    raxmlng = RAxMLNG(RAXML_NG_EXE)
    msa = MSA(msa_path)
    model = "WAG+G" if get_msa_type(msa_path)=="AA" else "GTR+G"
    msa_features = get_all_features(raxmlng, msa, model)
    difficulty = predictor.predict(msa_features)
    return difficulty


def get_msa_stats(msa_path):
    msa_data = get_alignment_data(msa_path)
    n_seq, n_loci = len(msa_data), len(msa_data[0].seq)
    all_msa_features = {"feature_n_seq": n_seq, "feature_n_loci": n_loci, "msa_path": msa_path,
                        "msa_name": get_msa_name(msa_path, GENERAL_MSA_DIR), "feature_msa_type": get_msa_type(msa_path)}
    alignment_data = get_alignment_data(msa_path)
    alignment_df = alignment_list_to_df(alignment_data)
    alignment_df_fixed = alignment_df.replace('-', np.nan)
    gap_lengths = alignment_df_fixed.isna().sum(axis=1)
    gap_pct = alignment_df_fixed.isna().sum(axis=1) / n_loci
    alignment_df_unique = alignment_df.T.drop_duplicates().T
    gaps = (alignment_df_fixed.isnull().sum() / len(alignment_df_fixed))
    counts_per_position = [dict(alignment_df_fixed[col].value_counts(dropna=True)) for col in list(alignment_df)]
    probabilities = [list(map(lambda x: x / sum(counts_per_position[col].values()), counts_per_position[col].values()))
                     for col in
                     list(alignment_df)]
    entropy = [sum(list(map(lambda x: -x * np.log(x), probabilities[col]))) for col in list(alignment_df)]
    constant_sites_pct = sum([1 for et in entropy if et == 0]) / len(entropy)
    msa_difficulty = pypythia(msa_path)
    all_msa_features.update({"feature_constant_sites_pct": constant_sites_pct, "feature_avg_entropy": np.mean(entropy),
                             "feature_25_pct_entropy": np.percentile(entropy, 25),
                             "feature_75_pct_entropy": np.percentile(entropy, 75),
                             "feature_median_entropy": np.median(entropy),
                             "feature_mean_gap_positions_pct": np.mean(gaps),
                             "feature_25_pct_gaps": np.percentile(gaps, 25),
                             "feature_75_pct_gaps": np.percentile(gaps, 75),
                             "feature_median_gaps": np.median(gaps),
                             "feature_gap_var": np.var(gap_lengths),
                             "feature_gap_max_by_min": ((np.max(gap_lengths) - np.min(gap_lengths)) / (
                                     np.max(gap_lengths) + 0.00001)), "feature_min_gap_pct": np.min(gap_pct),
                             "feature_n_unique_sites": len(alignment_df_unique.columns),
                             "feature_frac_unique_sites": len(alignment_df_unique.columns) / len(alignment_df.columns),
                            "pypythia_msa_difficulty": msa_difficulty}
                            )
    return all_msa_features


def tree_metrics(curr_run_directory,all_parsimony_trees,tree_object):
    tmp_folder = os.path.join(curr_run_directory, 'tmp_working_dir_tree_metrics')
    create_or_clean_dir(tmp_folder)
    curr_tree_path = os.path.join(tmp_folder,"trees_path")
    with open(curr_tree_path,'w') as TREE:
            TREE.write(tree_object.write(format=1))
    BL_metrics = tree_branch_length_metrics(tree_object)
    tree_distances = get_distances_between_leaves(tree_object)
    rf_values = []
    for parsimony_tree in all_parsimony_trees:
        rf_values.append(rf_distance(curr_run_directory,tree_object.write(format=1),parsimony_tree, name = "tree_vs_parsimony_rf"))
    res = {
        "feature_mean_branch_length": np.mean(BL_metrics["BL_list"]),
        "feature_mean_internal_branch_length": np.mean(BL_metrics["internal_BL_list"]),
        "feature_mean_leaf_branch_length": np.mean(BL_metrics["leaf_BL_list"]),
        "feature_tree_MAD": mad_tree_parameter(curr_tree_path),
        "feature_largest_branch_length": np.max(BL_metrics["BL_list"]),
        "feature_minimal_branch_length": np.min(BL_metrics["BL_list"]),
        "feature_median_branch_length": np.median(BL_metrics["BL_list"]),
        "feature_25_pct_branch_length": np.percentile(BL_metrics["BL_list"], 25),
        "feature_75_pct_branch_length": np.percentile(BL_metrics["BL_list"], 75),
        "feature_largest_distance_between_taxa": np.max(tree_distances),
        "feature_smallest_distance_between_taxa": np.min(tree_distances),
        "feature_25_pct_distance_between_taxa": np.percentile(tree_distances, 25),
        "feature_75_pct_distance_between_taxa": np.percentile(tree_distances, 75),
        'feature_mean_rf_distance': np.mean(rf_values), 'feature_max_rf_distance': np.max(rf_values),
        'feature_min_rf_distance': np.min([d for d in rf_values if d>0]),
        'feature_25_pct_rf_distance': np.percentile(rf_values, 25),
        'feature_75_pct_rf_distance': np.percentile(rf_values, 75),
        'feature_median_rf_relative_distance': np.median(rf_values)

    }
    return res


def get_local_path(path):
    if LOCAL_RUN:
        return path.replace("/groups/pupko/noaeker/", "/Users/noa/Workspace/")
    else:
        return path


def msa_features_pipeline(msa_path, existing_msa_features_path):
    if os.path.exists(existing_msa_features_path):
        logging.info("Using existing MSA general features")
        existing_msa_features = pickle.load(open(existing_msa_features_path, "rb"))
        return existing_msa_features
    logging.info("Calculating MSA general features from beggining")
    msa_general_features = pd.DataFrame.from_dict(
        {msa_path: get_msa_stats(get_local_path(msa_path))}, orient='index')
    pickle.dump(msa_general_features, open(existing_msa_features_path, 'wb'))
    return msa_general_features


def tree_features_pipeline(msa_path, curr_run_directory, msa_raw_data, existing_tree_features_path, iterations):
    tmp_folder = os.path.join(curr_run_directory, 'tmp_working_dir_tree_features_pipeline')
    create_or_clean_dir(tmp_folder)
    if os.path.exists(existing_tree_features_path):
        existing_tree_features = pickle.load(open(existing_tree_features_path, "rb"))
        return existing_tree_features
    trees_features_data = msa_raw_data[
        ["starting_tree_ind", "starting_tree_type", "starting_tree_object"]].drop_duplicates().reset_index()
    tree_features_dict = {}
    all_parsimony_trees = trees_features_data[trees_features_data["starting_tree_type"] == "pars"][
        "starting_tree_object"].drop_duplicates().tolist()
    for index, row in trees_features_data.iterrows():
        optimized_tree_object_ll, optimized_tree_object_alpha, optimized_tree_object = EVAL_tree_object_ll(
            generate_tree_object_from_newick(row["starting_tree_object"]), tmp_folder, get_local_path(msa_path),
            get_msa_type(get_local_path(msa_path)), opt_brlen=True)
        basic_features = {"feature_optimized_ll": optimized_tree_object_ll,
                                   'feature_optimized_tree_object_alpha': optimized_tree_object_alpha}
        distances, ll_improvements = get_random_spr_moves_vs_distances(optimized_tree_object,optimized_tree_object_ll, 20,tmp_folder, get_local_path(msa_path), get_msa_type(get_local_path(msa_path)))
        SPR_feature_metrics = {'feature_median_ll_improvement': np.median(ll_improvements),
                               'feature_median_ll_improvement': np.median(ll_improvements),
                               '75_pct_ll_improvement': np.percentile(ll_improvements,75),
                               'feature_25_pct_ll_improvement': np.percentile(ll_improvements, 25),
                               'feature_max_ll_improvement' : np.max(ll_improvements),
                               'feature_max_ll_improvement_radius': distances[np.argmax(ll_improvements)],
                               'feature_min_ll_improvement': np.min(ll_improvements),
                               'feature_min_ll_improvement_radius': distances[np.argmin(ll_improvements)]
                               }
        curr_tree_features = tree_metrics(tmp_folder,all_parsimony_trees, optimized_tree_object)
        curr_tree_features.update(basic_features)
        curr_tree_features.update(SPR_feature_metrics)
        tree_features_dict[(row["starting_tree_ind"], row["starting_tree_type"])] = curr_tree_features
    tree_features = pd.DataFrame.from_dict(tree_features_dict, orient='index')
    pickle.dump(tree_features, open(existing_tree_features_path, 'wb'))
    return tree_features


def process_all_msa_RAxML_runs(curr_run_directory, given_msa_data):
    '''

    :param curr_run_directory:
    :param given_msa_data:
    :return:
    '''
    best_msa_ll = max(given_msa_data["final_ll"])
    given_msa_data["best_msa_ll"] = best_msa_ll
    best_msa_tree_topology = max(given_msa_data[given_msa_data["final_ll"] == best_msa_ll]['final_tree_topology'])
    given_msa_data["rf_from_overall_msa_best_topology"] = given_msa_data["final_tree_topology"].apply(
        lambda x: rf_distance(curr_run_directory, x, best_msa_tree_topology, name = "MSA_enrichment_RF_calculations"))
    given_msa_data["delta_ll_from_overall_msa_best_topology"] = np.where(
        (given_msa_data["rf_from_overall_msa_best_topology"]) > 0, best_msa_ll - given_msa_data["final_ll"], 0)
    return given_msa_data


def enrich_raw_data(curr_run_directory, raw_data, iterations):
    '''

    :param curr_run_directory:
    :param raw_data:
    :return:  Enrich raw_data to estimate best LL for each MSA
    '''
    enriched_datasets = []
    MSAs = raw_data["msa_path"].unique()
    logging.info(f"Number of MSAs to work on: {len(MSAs)}")
    for i, msa in enumerate(MSAs):
        logging.info(f"Working on MSA number {i} in : {msa}")
        msa_folder = os.path.join(curr_run_directory, get_msa_name(msa, GENERAL_MSA_DIR))
        create_dir_if_not_exists(msa_folder)
        existing_msa_features_path = os.path.join(msa_folder, "msa_features")
        existing_tree_features_path = os.path.join(msa_folder, "tree_features")
        msa_final_dataset_path = os.path.join(msa_folder, "final_dataset")
        if os.path.exists(msa_final_dataset_path):
            logging.info("## Using existing data features for this MSA")
            msa_enriched_data = pickle.load(open(msa_final_dataset_path, "rb"))
        else:
            logging.info("## Calculating features from beggining for this MSA")
            msa_data = raw_data[raw_data["msa_path"] == msa].copy()
            msa_enriched_data = process_all_msa_RAxML_runs(msa_folder, msa_data)
            msa_features = msa_features_pipeline(msa, existing_msa_features_path)
            logging.info(f"MSA features: {msa_features}")
            msa_enriched_data = msa_enriched_data.merge(msa_features, right_index=True, left_on=["msa_path"])
            tree_features = tree_features_pipeline(msa, msa_folder, msa_data, existing_tree_features_path, iterations)
            msa_enriched_data = msa_enriched_data.merge(tree_features, right_index=True,
                                                        left_on=["starting_tree_ind", "starting_tree_type"])
            pickle.dump(msa_enriched_data, open(msa_final_dataset_path, "wb"))
        enriched_datasets.append(msa_enriched_data)
    enriched_data = pd.concat(enriched_datasets)
    logging.info(f"Number of unique MSAs in final result is {len(enriched_data['msa_path'].unique())}")
    return enriched_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_data_path', action='store', type=str,
                        default=f"/Users/noa/Workspace/raxml_deep_learning_results/local_data_generation/current_raw_results/global_csv_new.tsv")
    parser.add_argument('--features_out_path', action='store', type=str,
                        default=f"/Users/noa/Workspace/raxml_deep_learning_results/local_data_generation/current_ML_results/features{CSV_SUFFIX}")
    parser.add_argument('--results_folder', action='store', type=str,
                        default=RESULTS_FOLDER)
    parser.add_argument('--min_n_observations', action='store', type=int, default=1240)
    parser.add_argument('--iterations', action='store', type=int, default=40)
    args = parser.parse_args()
    curr_run_directory = os.path.join(args.results_folder, "features_extraction_pipeline_files")
    create_dir_if_not_exists(curr_run_directory)
    log_file_path = os.path.join(curr_run_directory, "features.log")
    logging.basicConfig(filename=log_file_path, level=logging.INFO)
    raw_data = pd.read_csv(args.raw_data_path, sep=CSV_SEP)
    counts = raw_data['msa_path'].value_counts()
    idx = counts[counts < args.min_n_observations].index
    raw_data = raw_data[~raw_data['msa_path'].isin(idx)]
    if LOCAL_RUN:
        np.random.seed(SEED)
        msa_names = list(np.unique(raw_data["msa_path"]))
        msas_sample = np.random.choice(msa_names, size=3, replace=False)
        raw_data = raw_data[raw_data["msa_path"].isin(msas_sample)]
    raw_data_with_features = enrich_raw_data(curr_run_directory, raw_data, iterations=args.iterations)
    logging.info(f'Writing enriched data to {args.features_out_path}')
    raw_data_with_features.to_csv(args.features_out_path, sep=CSV_SEP)


if __name__ == "__main__":
    main()
