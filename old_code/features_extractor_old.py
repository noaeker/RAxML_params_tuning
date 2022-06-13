from side_code.SPR_moves import *
from side_code.raxml import *
from side_code.basic_trees_manipulation import *
from side_code.MSA_manipulation import get_alignment_data, get_msa_name
from side_code.file_handling import create_or_clean_dir, create_dir_if_not_exists
from side_code.MSA_manipulation import get_alignment_data,alignment_list_to_df
from side_code.config import *
import pandas as pd
import pickle


def get_positions_stats(msa_path):
    alignment_data = get_alignment_data(msa_path)
    alignment_df = alignment_list_to_df(alignment_data)
    alignment_df_fixed = alignment_df.replace('-', np.nan)
    gap_positions_pct = np.mean(alignment_df_fixed.isnull().sum() / len(alignment_df_fixed))
    counts_per_position = [dict(alignment_df_fixed[col].value_counts(dropna=True)) for col in list(alignment_df)]
    probabilities = [list(map(lambda x: x / sum(counts_per_position[col].values()), counts_per_position[col].values()))
                     for col in
                     list(alignment_df)]
    entropy = [sum(list(map(lambda x: -x * np.log(x), probabilities[col]))) for col in list(alignment_df)]
    avg_entropy = np.mean(entropy)
    constant_sites_pct = sum([1 for et in entropy if et == 0]) / len(entropy)
    return {"feature_constant_sites_pct":constant_sites_pct,"feature_avg_entropy": avg_entropy, "feature_gap_positions_pct":gap_positions_pct}


def extract_features(msa_path, curr_run_directory,existing_features_dir, i):
    msa_features_path = os.path.join(existing_features_dir, msa_path.replace('/','_'))
    create_dir_if_not_exists(msa_features_path)
    msa_data = get_alignment_data(msa_path)
    n_seq, n_loci = len(msa_data), len(msa_data[0].seq)
    msa_path_no_extension = os.path.splitext(msa_path)[0]
    if re.search('\w+D[\da-z]+', msa_path_no_extension.split(os.sep)[-2]) is not None:
        msa_type = "DNA"
    else:
        msa_type = "AA"
    all_features = {"n_seq": n_seq, "n_loci": n_loci, "msa_path": msa_path,
                    "msa_name": get_msa_name(msa_path, GENERAL_MSA_DIR)}
    general_raxml_folder = os.path.join(curr_run_directory, "general_raxml_features")
    os.mkdir(general_raxml_folder)
    features_path = os.path.join(msa_features_path,"tree_features")
    raxml_statistics = extract_raxml_statistics_from_msa(msa_path=msa_path, msa_type = msa_type,curr_run_directory =curr_run_directory,output_name = "MSA_features")
    all_features.update(raxml_statistics)
    positions_stats_dict = get_positions_stats(msa_path = msa_path)
    all_features.update(positions_stats_dict)
    if os.path.exists(features_path):
        trees_data = pickle.load(open(features_path,"rb"))
        parsimony_trees_ll_on_data = trees_data["parsimony_trees_ll_on_data"]
        parsimony_trees_path = trees_data ["parsimony_trees_path"]
        random_trees_ll_on_data = trees_data["random_trees_ll_on_data"]
        random_trees_path = trees_data ["random_trees_path"]
        parsimony_rf_distances = trees_data["parsimony_rf_distances"]
    else:
        several_parsimony_and_random_folder = os.path.join(curr_run_directory, f"parsimony_and_random_statistics_{i}")
        os.mkdir(several_parsimony_and_random_folder)
        os.mkdir(several_parsimony_and_random_folder)
        os.mkdir(several_parsimony_and_random_folder)
        parsimony_trees_path = generate_n_tree_topologies(30, msa_path, several_parsimony_and_random_folder,
                                                          seed=SEED, tree_type="parsimony", msa_type=msa_type)

        parsimony_trees_ll_on_data, parsimony_trees_path = raxml_optimize_trees_for_given_msa(msa_path,
                                                                                              f"{i}_parsimony_eval",
                                                                                              parsimony_trees_path,
                                                                                              several_parsimony_and_random_folder,
                                                                                              msa_type, opt_brlen=True
                                                                                              )
        local_parsimony_path = f'{msa_features_path}/parsimony'
        shutil.copy(parsimony_trees_path,local_parsimony_path)

        random_trees_path = generate_n_tree_topologies(30, msa_path, several_parsimony_and_random_folder,
                                                       seed=SEED, tree_type="random", msa_type=msa_type)
        random_trees_ll_on_data, random_trees_path = raxml_optimize_trees_for_given_msa(msa_path, f"{i}_random_eval",
                                                                                        random_trees_path,
                                                                                        several_parsimony_and_random_folder,
                                                                                        msa_type, opt_brlen=True
                                                                                        )

        local_random_path = f'{msa_features_path}/random'
        shutil.copy(random_trees_path, local_random_path)
        parsimony_rf_distances = np.array(RF_distances(curr_run_directory, parsimony_trees_path))

        trees_data = {"parsimony_trees_ll_on_data" : parsimony_trees_ll_on_data, "parsimony_trees_path": local_parsimony_path,"parsimony_rf_distances":parsimony_rf_distances,
                                "random_trees_ll_on_data" :random_trees_ll_on_data,"random_trees_path" :local_random_path,

                                }

        pickle.dump(trees_data,open(features_path,'wb'))

    parsimony_tree_objects = generate_multiple_tree_object_from_newick(parsimony_trees_path)
    tree_features_dict = {'feature_avg_tree_divergence': np.mean([compute_tree_divergence(parsimony_tree) for parsimony_tree in parsimony_tree_objects]),
                          'feature_var_tree_divergence': np.var(
                              [compute_tree_divergence(parsimony_tree) for parsimony_tree in parsimony_tree_objects]),
                          'feature_avg_largest_branch_length': np.mean([compute_largest_branch_length(parsimony_tree) for parsimony_tree in parsimony_tree_objects]),
                          'feature_var_largest_branch_length': np.var(
                              [compute_largest_branch_length(parsimony_tree) for parsimony_tree in
                               parsimony_tree_objects]),
                          'feature_avg_largest_distance_between_taxa': np.mean([max_distance_between_leaves(parsimony_tree) for parsimony_tree in parsimony_tree_objects]),
                          'feature_avg_tree_MAD': np.mean([mad_tree_parameter(curr_run_directory,parsimony_tree) for parsimony_tree in parsimony_tree_objects]),
                          'feature_avg_parsimony_rf_dist': np.mean(parsimony_rf_distances),
                          'feature_mean_unique_topolgies_rf_dist': np.mean(parsimony_rf_distances>0),
                          'feature_max_parsimony_rf_dist': np.max(parsimony_rf_distances),
                          'feature_best_parsimony_vs_best_random': (
                                      max(parsimony_trees_ll_on_data) - max(random_trees_ll_on_data)),
                          'feature_worse_parsimony_vs_best_random': (
                                  min(parsimony_trees_ll_on_data) - max(random_trees_ll_on_data)),
                          'feature_best_parsimony_vs_worse_random': (
                                  max(parsimony_trees_ll_on_data) - min(random_trees_ll_on_data)),
                          'feature_parsimony_ll_var_vs_random_ll_var': (
                                  np.var(parsimony_trees_ll_on_data) / np.var(random_trees_ll_on_data)),
                          'feature_mean_parsimony_scores' :max(parsimony_trees_ll_on_data)-np.mean(random_trees_ll_on_data)
                          #'distances_vs_ll_corr': np.corrcoef(np.array(distances), np.array(ll_improvements))[0, 1],

                          }

    all_features.update(tree_features_dict)
    return all_features


def main():
    overall_data_path = f"/Users/noa/Workspace/raxml_deep_learning_results/current_raw_results/global_csv_enriched_agg.tsv"
    out_path = f"/Users/noa/Workspace/raxml_deep_learning_results/current_ML_results/features{CSV_SUFFIX}"
    curr_run_directory = os.path.join(RESULTS_FOLDER, "features_extraction_test")
    existing_features_dir = os.path.join(RESULTS_FOLDER, "features_per_msa_dump")
    create_dir_if_not_exists(existing_features_dir)
    data = pd.read_csv(overall_data_path, sep=CSV_SEP)
    msa_paths = list(np.unique(data["msa_path"]))
    if LOCAL_RUN:
        msa_paths = [msa_path.replace("/groups/pupko/noaeker/", "/Users/noa/Workspace/") for msa_path in msa_paths]
        msa_paths = [msa_path.replace("/groups/pupko/noaeker/", "/Users/noa/Workspace/") for msa_path in msa_paths]
    results = []
    for i, msa_path in enumerate(msa_paths):
        create_or_clean_dir(curr_run_directory)
        msa_features = extract_features(msa_path,curr_run_directory,existing_features_dir, i)
        results.append(msa_features)
    results_df = pd.DataFrame(results)
    results_df.to_csv(out_path, sep=CSV_SEP)


if __name__ == "__main__":
    main()
