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


def tree_metrics(curr_run_directory,starting_tree_str):
    tree_object = generate_tree_object_from_newick(starting_tree_str)
    res = {"tree_divergence": compute_tree_divergence(tree_object),
        "tree_MAD" : mad_tree_parameter(curr_run_directory,tree_object),
           "largest_branch_length": compute_largest_branch_length(tree_object),
           "largest_distance_between_taxa": max_distance_between_leaves(tree_object),
           "largest_distance_between_taxa": max_distance_between_leaves(tree_object)

     }
    return res


def main():
    raw_data_path = f"/Users/noa/Workspace/raxml_deep_learning_results/current_raw_results/global_csv_enriched_new.tsv"
    out_path = f"/Users/noa/Workspace/raxml_deep_learning_results/current_ML_results/features{CSV_SUFFIX}"
    curr_run_directory = os.path.join(RESULTS_FOLDER, "features_extraction")
    existing_features_dir = os.path.join(RESULTS_FOLDER, "features_per_msa_dump")
    create_dir_if_not_exists(existing_features_dir)
    raw_data = pd.read_csv(raw_data_path, sep=CSV_SEP)
    raw_data = raw_data.sample(n=10)
    raw_data["local_msa_path"] = raw_data["msa_path"].apply(lambda x: x.replace("/groups/pupko/noaeker/", "/Users/noa/Workspace/"))
    #msas_features = pd.DataFrame.from_dict({msa: get_positions_stats(msa) for msa in np.unique(raw_data["local_msa_path"])})
    trees_features_data = raw_data[["msa_path","starting_tree_ind","starting_tree_object"]].drop_duplicates().reset_index()
    tree_features = {(row["msa_path"]+str(row["starting_tree_ind"])) : tree_metrics(curr_run_directory,row["starting_tree_object"]) for index,row in trees_features_data.iterrows()}
    pickle.dump(tree_features, open(features_path, 'wb'))
    print(tree_features)

if __name__ == "__main__":
    main()
