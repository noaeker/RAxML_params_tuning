from side_code.SPR_moves import *
from side_code.raxml import *
from side_code.basic_trees_manipulation import *
from side_code.MSA_manipulation import get_msa_data, get_msa_name
from side_code.file_handling import create_or_clean_dir, create_dir_if_not_exists
from side_code.config import *
import pandas as pd
import pickle


def extract_features(msa_path, curr_run_directory,existing_features_dir, i):
    msa_features_path = os.path.join(existing_features_dir, f'msa_{i}')
    create_dir_if_not_exists(msa_features_path)
    msa_data = get_msa_data(msa_path, 'fasta')
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
    if os.path.exists(features_path):
        trees_data = pickle.load(open(features_path,"wb"))
        parsimony_trees_ll_on_data = trees_data ["parsimony_trees_ll_on_data"]
        parsimony_trees_path = trees_data ["parsimony_trees_path"]
        random_trees_ll_on_data = trees_data["random_trees_ll_on_data"]
        random_trees_path = trees_data ["random_trees_path"]
    else:
        several_parsimony_and_random_folder = os.path.join(curr_run_directory, f"parsimony_and_random_statistics_{i}")
        os.mkdir(several_parsimony_and_random_folder)
        parsimony_trees_path = generate_n_tree_topologies(30, msa_path, several_parsimony_and_random_folder,
                                                          seed=SEED, tree_type="parsimony", msa_type=msa_type)

        parsimony_trees_ll_on_data, parsimony_trees_path = raxml_optimize_trees_for_given_msa(msa_path,
                                                                                              f"{i}_parsimony_eval",
                                                                                              parsimony_trees_path,
                                                                                              several_parsimony_and_random_folder,
                                                                                              msa_type, opt_brlen=True
                                                                                              )

        random_trees_path = generate_n_tree_topologies(30, msa_path, several_parsimony_and_random_folder,
                                                       seed=SEED, tree_type="random", msa_type=msa_type)
        random_trees_ll_on_data, random_trees_path = raxml_optimize_trees_for_given_msa(msa_path, f"{i}_random_eval",
                                                                                        random_trees_path,
                                                                                        several_parsimony_and_random_folder,
                                                                                        msa_type, opt_brlen=True
                                                                                        )


        trees_data = {"parsimony_trees_ll_on_data" : parsimony_trees_ll_on_data, "parsimony_trees_path": parsimony_trees_path,
                                "random_trees_ll_on_data" :random_trees_ll_on_data,"random_trees_path" :random_trees_path
                                }
        pickle.dump(trees_data,open(features_path,'wb'))

    parsimony_tree_objects = generate_multiple_tree_object_from_newick(parsimony_trees_path)

    parsimony_rf_distances = np.array(RF_distances(curr_run_directory, parsimony_trees_path))
    tree_features_dict = {'avg_tree_divergence': np.mean([compute_tree_divergence(parsimony_tree) for parsimony_tree in parsimony_tree_objects]),
                          'var_tree_divergence': np.var(
                              [compute_tree_divergence(parsimony_tree) for parsimony_tree in parsimony_tree_objects])
                          'avg_largest_branch_length': np.mean([compute_largest_branch_length(parsimony_tree) for parsimony_tree in parsimony_tree_objects]),
                          'var_largest_branch_length': np.var(
                              [compute_largest_branch_length(parsimony_tree) for parsimony_tree in
                               parsimony_tree_objects]),
                          'avg_largest_distance_between_taxa': np.mean([max_distance_between_leaves(parsimony_tree) for parsimony_tree in parsimony_tree_objects]),
                          'avg_tree_MAD': np.mean([mad_tree_parameter(parsimony_tree) for parsimony_tree in parsimony_tree_objects]),
                          'avg_parsimony_rf_dist': np.mean(parsimony_rf_distances),
                          'mean_unique_topolgies_rf_dist': np.mean(parsimony_rf_distances>0),
                          'max_parsimony_rf_dist': np.max(parsimony_rf_distances),
                          'best_parsimony_vs_best_random': (
                                      max(parsimony_trees_ll_on_data) - max(random_trees_ll_on_data)),
                          'worse_parsimony_vs_best_random': (
                                  min(parsimony_trees_ll_on_data) - max(random_trees_ll_on_data)),
                          'best_parsimony_vs_worse_random': (
                                  max(parsimony_trees_ll_on_data) - min(random_trees_ll_on_data)),
                          'parsimony_ll_var_vs_random_ll_var': (
                                  np.var(parsimony_trees_ll_on_data) / np.var(random_trees_ll_on_data)),
                          #'distances_vs_ll_corr': np.corrcoef(np.array(distances), np.array(ll_improvements))[0, 1],

                          }
    all_features.update(tree_features_dict)
    return all_features


def main():
    overall_data_path = f"{DATASETS_FOLDER}/full_raxml_data{CSV_SUFFIX}"
    out_path = f"{ML_RESULTS_FOLDER}/features{CSV_SUFFIX}"
    curr_run_directory = os.path.join(RESULTS_FOLDER, "features_extraction_test")
    existing_features_dir = os.path.join(RESULTS_FOLDER, "features_per_msa_dump")
    create_dir_if_not_exists(existing_features_dir)
    data = pd.read_csv(overall_data_path, sep=CSV_SEP)
    msa_paths = list(np.unique(data["original_alignment_path"]))
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
