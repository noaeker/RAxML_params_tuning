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
from side_code.MSA_manipulation import get_alignment_data, alignment_list_to_df
from side_code.config import *
import pandas as pd
import pickle
import argparse


def get_msa_stats(msa_path):
    msa_data = get_alignment_data(msa_path)
    n_seq, n_loci = len(msa_data), len(msa_data[0].seq)
    msa_path_no_extension = os.path.splitext(msa_path)[0]
    if re.search('\w+D[\da-z]+', msa_path_no_extension.split(os.sep)[-2]) is not None:
        msa_type = "DNA"
    else:
        msa_type = "AA"
    all_msa_features = {"feature_n_seq": n_seq, "feature_n_loci": n_loci, "feature_msa_path": msa_path,
                    "feature_msa_name": get_msa_name(msa_path, GENERAL_MSA_DIR), "feature_msa_type": msa_type}
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
    all_msa_features.update({"feature_constant_sites_pct": constant_sites_pct, "feature_avg_entropy": avg_entropy,
            "feature_gap_positions_pct": gap_positions_pct})
    return all_msa_features


def tree_metrics(curr_run_directory, starting_tree_str):
    tree_object = generate_tree_object_from_newick(starting_tree_str)
    res = {"feature_tree_divergence": compute_tree_divergence(tree_object),
           "feature_tree_MAD": mad_tree_parameter(curr_run_directory, tree_object),
           "feature_largest_branch_length": compute_largest_branch_length(tree_object),
           "feature_largest_distance_between_taxa": max_distance_between_leaves(tree_object)

           }
    return res


def get_trees_file(curr_run_directory, raw_data, msa_path, starting_tree_type):
    tree_files_path = os.path.join(curr_run_directory, 'tmp_trees_file')
    trees = raw_data[((raw_data["msa_path"] == msa_path) & (raw_data["starting_tree_type"] == starting_tree_type))][
        "starting_tree_object"].tolist()
    with open(tree_files_path, 'w') as TREES_PATH:
        TREES_PATH.writelines(trees)
    return tree_files_path


def tree_group_metrics(curr_run_directory, raw_data, msa_path, starting_tree_type):
    trees_file_path = get_trees_file(curr_run_directory, raw_data, msa_path, starting_tree_type)
    rf_values = RF_distances(curr_run_directory, trees_file_path)
    res = {'feature_mean_rf_distance': np.mean(rf_values), 'feature_max_rf_distance' : np.max(rf_values)
           }

    return res


def get_local_path(path):
    if LOCAL_RUN:
        return path.replace("/groups/pupko/noaeker/", "/Users/noa/Workspace/")
    else:
        return path

def msa_features_pipeline(raw_data,existing_msa_features_path):
    if os.path.exists(existing_msa_features_path):
        existing_tree_features = pickle.load(open(existing_msa_features_path,"rb"))
        return existing_tree_features
    msa_positions_features = pd.DataFrame.from_dict(
        {msa: get_msa_stats(get_local_path(msa)) for msa in np.unique(raw_data["msa_path"])},orient ='index')
    pickle.dump(msa_positions_features, open(existing_msa_features_path, 'wb'))
    return msa_positions_features

def tree_features_pipeline(curr_run_directory,raw_data,existing_tree_features_path):
    if os.path.exists(existing_tree_features_path):
        existing_tree_features = pickle.load(open(existing_tree_features_path,"rb"))
        return existing_tree_features
    trees_features_data = raw_data[
        ["msa_path", "starting_tree_ind", "starting_tree_type","starting_tree_object"]].drop_duplicates().reset_index()
    tree_features = pd.DataFrame.from_dict({
        (row["msa_path"] , row["starting_tree_ind"],row["starting_tree_type"]): tree_metrics(curr_run_directory,
                                                                              row["starting_tree_object"])
        for index, row in trees_features_data.iterrows()}, orient= 'index')
    pickle.dump(tree_features, open(existing_tree_features_path, 'wb'))
    return tree_features

def tree_group_features_pipeline(curr_run_directory,raw_data,existing_tree_group_features_path):
    if os.path.exists(existing_tree_group_features_path):
        existing_tree_features = pickle.load(open(existing_tree_group_features_path,"rb"))
        return existing_tree_features
    tree_groups_data = raw_data[["msa_path", "starting_tree_type"]].drop_duplicates().reset_index()
    tree_group_features = pd.DataFrame.from_dict({
        (row["msa_path"] ,row["starting_tree_type"]): tree_group_metrics(curr_run_directory, raw_data,
                                                                                     row["msa_path"],
                                                                                     row["starting_tree_type"]) for
        index, row in tree_groups_data.iterrows()}, orient = 'index')
    pickle.dump(tree_group_features, open(existing_tree_group_features_path, 'wb'))
    return tree_group_features


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
        lambda x: rf_distance(curr_run_directory, x, best_msa_tree_topology))
    given_msa_data["delta_ll_from_overall_msa_best_topology"] = np.where(
        (given_msa_data["rf_from_overall_msa_best_topology"]) > 0,  best_msa_ll-given_msa_data["final_ll"], 0)
    return given_msa_data

def enrich_raw_data(curr_run_directory,raw_data):
    '''

    :param curr_run_directory:
    :param raw_data:
    :return:  Enrich raw_data to estimate best LL for each MSA
    '''
    enriched_datasets = []
    for msa in raw_data["msa_path"].unique():
        msa_data = raw_data[raw_data["msa_path"] == msa].copy()
        msa_enriched_data = process_all_msa_RAxML_runs(curr_run_directory, msa_data)
        enriched_datasets.append(msa_enriched_data)
    enriched_data = pd.concat(enriched_datasets)
    return enriched_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_data_path', action='store', type=str,
                        default=f"/Users/noa/Workspace/raxml_deep_learning_results/current_raw_results/global_csv_new.tsv")
    parser.add_argument('--features_out_path', action='store', type=str,
                        default=f"/Users/noa/Workspace/raxml_deep_learning_results/current_ML_results/features{CSV_SUFFIX}")
    parser.add_argument('--results_folder', action='store', type=str,
                        default=RESULTS_FOLDER)
    parser.add_argument('--min_n_observations', action='store', type=int, default = 1240)
    args = parser.parse_args()
    curr_run_directory = os.path.join(args.results_folder, "features_extraction_pipeline_files")
    create_dir_if_not_exists(curr_run_directory)
    existing_features_dir = os.path.join(args.results_folder, "features_per_msa_dump")
    create_dir_if_not_exists(existing_features_dir)
    existing_msa_features_path = os.path.join(existing_features_dir, "msa_features")
    existing_tree_features_path = os.path.join(existing_features_dir, "tree_features")
    existing_tree_group_features_path = os.path.join(existing_features_dir, "tree_group_features")
    raw_data = pd.read_csv(args.raw_data_path, sep=CSV_SEP)
    raw_data = enrich_raw_data(curr_run_directory,raw_data)
    counts = raw_data['msa_path'].value_counts()
    idx = counts[counts < args.min_n_observations].index
    raw_data = raw_data[~raw_data['msa_path'].isin(idx)]
    if LOCAL_RUN:
        np.random.seed(SEED)
        msa_names = list(np.unique(raw_data["msa_path"]))
        msas_sample = np.random.choice(msa_names, size=3, replace=False)
        raw_data = raw_data[raw_data["msa_path"].isin(msas_sample)]
    msa_features = msa_features_pipeline(raw_data, existing_msa_features_path)
    raw_data = raw_data.merge(msa_features, right_index= True, left_on = ["msa_path"])
    tree_features = tree_features_pipeline(curr_run_directory,raw_data,existing_tree_features_path)
    raw_data = raw_data.merge(tree_features, right_index=True, left_on=["msa_path", "starting_tree_ind","starting_tree_type"])
    tree_group_features = tree_group_features_pipeline(curr_run_directory, raw_data, existing_tree_group_features_path)
    raw_data = raw_data.merge(tree_group_features, right_index= True, left_on=["msa_path","starting_tree_type"])
    raw_data.to_csv(args.features_out_path, sep = CSV_SEP)





if __name__ == "__main__":
    main()
