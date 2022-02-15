from config import *
import pandas as pd
import numpy as np
from raxml import *
from feature_calculator import *

def extract_features(msa_path, msa_type,curr_run_directory,i):
    with open(msa_path) as original:
        msa_data = list(SeqIO.parse(original, 'fasta'))
    n_seq, n_loci = len(msa_data), len(msa_data[0].seq)
    all_features = {"n_seq": n_seq, "n_loci": n_loci, "msa_path": msa_path, "msa_name": get_msa_name(msa_path, GENERAL_MSA_DIR)}
    RAxML_metrics_dict = extract_raxml_statistics_from_msa(msa_path, msa_type, f"raxml_{i}", curr_run_directory)
    all_features.update(RAxML_metrics_dict)
    tree_path = RAxML_metrics_dict["parsimony_tree_path"]
    tree_object = generate_tree_object_from_newick( tree_path )
    tree_features_dict = {'tree_divergence': compute_tree_divergence(tree_object),
                     'largest_branch_length': compute_largest_branch_length(tree_object),
                     'largest_distance_between_taxa': max_distance_between_leaves(tree_object),
                          'tree_MAD': mad_tree_parameter(tree_path)
                     }
    all_features.update(tree_features_dict)
    return all_features


def main():
    overall_data_path = f"{RESULTS_FOLDER}/full_raxml_data{CSV_SUFFIX}"
    out_path = f"{RESULTS_FOLDER}/features{CSV_SUFFIX}"
    curr_run_directory = os.path.join(RESULTS_FOLDER,"features_extraction")
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