
from side_code.SPR_moves import *
from side_code.raxml import *
from side_code.basic_trees_manipulation import *
from side_code.MSA_manipulation import get_msa_data, get_msa_name
from side_code.file_handling import create_or_clean_dir
from side_code.config import *
import pandas as pd



def extract_features(msa_path, msa_type,curr_run_directory,i):
    msa_data = get_msa_data(msa_path, 'fasta')
    n_seq, n_loci = len(msa_data), len(msa_data[0].seq)
    all_features = {"n_seq": n_seq, "n_loci": n_loci, "msa_path": msa_path, "msa_name": get_msa_name(msa_path, GENERAL_MSA_DIR)}
    general_raxml_folder = os.path.join(curr_run_directory,"general_raxml_features")
    os.mkdir(general_raxml_folder)
    RAxML_metrics_dict = extract_raxml_statistics_from_msa(msa_path, msa_type, f"raxml_{i}", general_raxml_folder)
    all_features.update(RAxML_metrics_dict)
    single_parsimony_tree_path = RAxML_metrics_dict["parsimony_tree_path"]
    single_parsimony_tree_obj = generate_tree_object_from_newick(single_parsimony_tree_path)
    several_parsimony_and_random_folder = os.path.join(curr_run_directory, f"parsimony_and_random_statistics_{i}")
    os.mkdir(several_parsimony_and_random_folder)
    parsimony_trees_path = generate_n_tree_topologies(30, msa_path, several_parsimony_and_random_folder,
                                                      seed  =SEED, tree_type = "parsimony", msa_type = msa_type)

    parsimony_trees_ll_on_data, parsimony_tree_paths = raxml_optimize_trees_for_given_msa(msa_path, f"{i}_parsimony_eval", parsimony_trees_path,
                                       several_parsimony_and_random_folder,  msa_type, opt_brlen=False
                                       )

    random_trees_path = generate_n_tree_topologies(30, msa_path, several_parsimony_and_random_folder,
                                                   seed  =SEED, tree_type = "random", msa_type = msa_type)
    random_trees_ll_on_data, random_tree_paths = raxml_optimize_trees_for_given_msa(msa_path, f"{i}_random_eval", random_trees_path,
                                       several_parsimony_and_random_folder,  msa_type, opt_brlen=False
                                       )

    distances, ll_improvements = get_random_spr_moves_vs_distances(single_parsimony_tree_path, 100, curr_run_directory, msa_path, msa_type)

    tree_features_dict = {'tree_divergence': compute_tree_divergence( single_parsimony_tree_obj ),
                     'largest_branch_length': compute_largest_branch_length( single_parsimony_tree_obj ),
                     'largest_distance_between_taxa': max_distance_between_leaves( single_parsimony_tree_obj ),
                          'tree_MAD': mad_tree_parameter( single_parsimony_tree_path),
                          'avg_parsimony_rf_dist': RF_distances(curr_run_directory, parsimony_trees_path),
                          'parsimony_vs_random_diff' : np.mean(parsimony_trees_ll_on_data)/ np.mean(random_trees_ll_on_data),
                          'parsimony_var_vs_mean' : np.var(parsimony_trees_ll_on_data)/ np.mean(parsimony_trees_ll_on_data),
                          'random_var_vs_mean': np.var(random_trees_ll_on_data)/ np.mean(random_trees_ll_on_data),
                          'best_parsimony_vs_best_random': (max(parsimony_trees_ll_on_data)/max(random_trees_ll_on_data)),
                          'distances_vs_ll_corr' : np.corrcoef(np.array(distances),np.array(ll_improvements))[0,1],



                     }
    all_features.update(tree_features_dict)
    return all_features


def main():
    overall_data_path = f"{RESULTS_FOLDER}/full_raxml_data{CSV_SUFFIX}"
    out_path = f"{RESULTS_FOLDER}/features{CSV_SUFFIX}"
    curr_run_directory = os.path.join(RESULTS_FOLDER,"features_extraction_test")
    data = pd.read_csv(overall_data_path, sep=CSV_SEP)
    msa_paths = list(np.unique(data["original_alignment_path"]))
    if LOCAL_RUN:
        msa_paths = [msa_path.replace("/groups/pupko/noaeker/","/Users/noa/Workspace/") for msa_path in msa_paths]
        msa_paths = [msa_path.replace("/groups/pupko/noaeker/", "/Users/noa/Workspace/") for msa_path in msa_paths]
    results = []
    for i,msa_path in enumerate(msa_paths):
        create_or_clean_dir(curr_run_directory)
        msa_path_no_extension = os.path.splitext(msa_path)[0]
        if re.search('\w+D[\da-z]+', msa_path_no_extension.split(os.sep)[-2]) is not None:
            msa_type = "DNA"
        else:
            msa_type = "AA"
        msa_raxml_features  = extract_features(msa_path, msa_type,curr_run_directory,i)
        results.append(msa_raxml_features)
    results_df = pd.DataFrame(results)
    results_df.to_csv(out_path, sep = CSV_SEP)


if __name__ == "__main__":
    main()