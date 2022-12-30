import sys
import math

if sys.platform == "linux" or sys.platform == "linux2":

    PROJECT_ROOT_DIRECRTORY = "/groups/pupko/noaeker/RAxML_params_tuning"
else:
    PROJECT_ROOT_DIRECRTORY = "/Users/noa/Workspace/RAxML_params_tuning"
sys.path.append(PROJECT_ROOT_DIRECRTORY)

from side_code.SPR_moves import *
from scipy.stats import spearmanr
from sklearn.decomposition import KernelPCA
from side_code.raxml import *
from side_code.basic_trees_manipulation import *
from side_code.MSA_manipulation import get_alignment_data, get_msa_name
from side_code.file_handling import create_or_clean_dir, create_dir_if_not_exists
from side_code.MSA_manipulation import get_alignment_data, alignment_list_to_df, get_msa_name, \
    get_local_path
from side_code.config import *
from ML_pipeline.features_job_functions import feature_job_parser
from sklearn.manifold import MDS, Isomap, TSNE, LocallyLinearEmbedding
from sklearn.decomposition import PCA
from shutil import rmtree
from sklearn.linear_model import LassoCV
import pandas as pd
import pickle
import argparse
from pypythia.predictor import DifficultyPredictor
from pypythia.prediction import get_all_features
from pypythia.raxmlng import RAxMLNG
from pypythia.msa import MSA
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import time


def pct_25(values):
    return np.percentile(values, 25)


def pct_75(values):
    return np.percentile(values, 75)


def max_vs_min(values):
    return (np.max(values) - np.min(values)) / (
            np.max(values) + 0.00001)


def get_summary_statistics_dict(feature_name, values, funcs={'mean': np.mean, 'var': np.var,
                                                             'pct_25': pct_25, 'pct_75': pct_75,
                                                             'max_vs_min': max_vs_min, 'min': np.min, 'max': np.max,
                                                             }):
    res = {}
    for func in funcs:
        res.update({f'{feature_name}_{func}': (funcs[func])(values)})
    return res


def pypythia(msa_path, args):
    predictor = DifficultyPredictor(open(f"{PROJECT_ROOT_DIRECRTORY}/pypythia/predictor.pckl", "rb"))
    raxmlng = RAxMLNG(RAXML_NG_EXE)
    msa = MSA(msa_path)
    model = "WAG+G" if args.msa_type == "AA" else "GTR+G"
    msa_features = get_all_features(raxmlng, msa, model)
    difficulty = predictor.predict(msa_features)
    return difficulty


def get_msa_stats(msa_path,args):
    msa_data = get_alignment_data(msa_path)
    n_seq, n_loci = len(msa_data), len(msa_data[0].seq)
    all_msa_features = {"feature_msa_n_seq": n_seq, "feature_msa_n_loci": n_loci, "msa_path": msa_path,
                        "msa_name": get_msa_name(msa_path, GENERAL_MSA_DIR), "feature_msa_type": args.msa_type}
    alignment_data = get_alignment_data(msa_path)
    alignment_df = alignment_list_to_df(alignment_data)
    alignment_df_fixed = alignment_df.replace('-', np.nan)
    gap_lengths = alignment_df_fixed.isna().sum(axis=1)/n_loci
    alignment_df_unique = alignment_df.T.drop_duplicates().T
    counts_per_position = [dict(alignment_df_fixed[col].value_counts(dropna=True)) for col in list(alignment_df)]
    probabilities = [list(map(lambda x: x / sum(counts_per_position[col].values()), counts_per_position[col].values()))
                     for col in
                     list(alignment_df)]
    entropy = [sum(list(map(lambda x: -x * np.log(x), probabilities[col]))) for col in list(alignment_df)]
    constant_sites_pct = sum([1 for et in entropy if et == 0]) / len(entropy)
    msa_difficulty = pypythia(msa_path,args)

    all_msa_features.update({"feature_msa_constant_sites_pct": constant_sites_pct,"feature_msa_n_unique_sites": len(alignment_df_unique.columns),"feature_frac_unique_sites": len(alignment_df_unique.columns) / len(alignment_df.columns),
                             "feature_msa_pypythia_msa_difficulty": msa_difficulty})

    multi_dimensional_features = {"feature_msa_entropy": entropy, "feature_msa_gap_positions_pct": gap_lengths}


    for feature in multi_dimensional_features:
         all_msa_features.update(get_summary_statistics_dict(feature, multi_dimensional_features[feature]))
    return all_msa_features



def msa_features_pipeline(msa_path, existing_msa_features_path, args):
    if os.path.exists(existing_msa_features_path):
        logging.info("Using existing MSA general features")
        existing_msa_features = pickle.load(open(existing_msa_features_path, "rb"))
        return existing_msa_features
    logging.info("Calculating MSA general features from beggining")
    msa_general_features = pd.DataFrame.from_dict(
        {msa_path: get_msa_stats(get_local_path(msa_path),args)}, orient='index').reset_index()
    msa_final_features = msa_general_features[[col for col in msa_general_features.columns if col.startswith("feature")]]
    msa_final_features["msa_path"] = msa_path
    pickle.dump(msa_final_features, open(existing_msa_features_path, 'wb'))
    return msa_final_features

def single_tree_metrics(curr_run_directory, all_parsimony_trees,all_parsimony_trees_LL, tree_object, tree_LL, msa_path, msa_type, spr_iterations = 50):
    tmp_folder = os.path.join(curr_run_directory, 'tmp_working_dir_tree_metrics')
    create_or_clean_dir(tmp_folder)
    distances, neighbors = get_random_spr_moves_vs_distances(tree_object, n_iterations=spr_iterations)
    SPR_trees_path = os.path.join(curr_run_directory, 'all_trees_SPR_neighbors')
    unify_text_files(neighbors, SPR_trees_path, str_given=True)
    # optimized_neighbor_ll, optimized_neighbor_alpha, optimized_neighbor_trees_file = raxml_optimize_trees_for_given_msa(
    #     get_local_path(msa_path), "trees_eval_SPR", SPR_trees_path,
    #     tmp_folder,msa_type, opt_brlen=False
    # )
    # ll_improvements = [neighbor_LL-tree_LL for neighbor_LL in optimized_neighbor_ll]

    #corcoeff_SPR, pval_SPR = spearmanr(ll_improvements, distances)
    curr_tree_path = os.path.join(tmp_folder, "trees_path")
    with open(curr_tree_path, 'w') as TREE:
        TREE.write(tree_object.write(format=1))
    BL_metrics = tree_branch_length_metrics(tree_object)
    tree_distances = get_distances_between_leaves(tree_object)
    #final_tree_distances = get_distances_between_leaves(final_tree_object)
    rf_values = []
    #parsimony_LL_differences = [tree_LL-pars_LL for pars_LL in all_parsimony_trees_LL]

    for parsimony_tree in all_parsimony_trees:
        rf_values.append(
            rf_distance(curr_run_directory, tree_object.write(format=1), parsimony_tree, name="tree_vs_parsimony_rf"))
    #
    # corcoeff, pval = spearmanr(parsimony_LL_differences, rf_values)
    # LL_rf_corr =  0 if math.isnan(corcoeff) else corcoeff
    affinity = [1-val for val in rf_values]
    if np.sum(affinity)==0:
        LL_neighbour_score = tree_LL
    else:
        LL_neighbour_score = np.dot(np.array(affinity), np.array(all_parsimony_trees_LL))/np.sum(affinity)

    all_tree_features = {"feature_tree_MAD": mad_tree_parameter(curr_tree_path), 'feature_tree_LL_neighbour_score': LL_neighbour_score, "tree_distances": tree_distances} #"final_tree_distances": final_tree_distances'feature_corcoeff_SPR': corcoeff_SPR ,'feature_tree_parsimony_dist_vs_LL_imprv_corr':LL_rf_corr,"feature_LL_diff_vs_parsimony": parsimony_LL_differences,'feature_ll_improvements': ll_improvements
    multidimensional_features = {'feature_tree_branch_lengths' : BL_metrics["BL_list"], "feature_tree_distances_between_taxa":tree_distances, "feature_tree_parsimony_rf_values" : rf_values}
    for feature in multidimensional_features:
        all_tree_features.update(get_summary_statistics_dict(feature, multidimensional_features[feature]))

    return all_tree_features, neighbors


def mds_and_pca(tree_features, X, starting_trees, mds_n_components, metric, pca_n_components, suffix):
    st = time.time()
    mds = MDS(random_state=0, n_components=mds_n_components, metric=True, dissimilarity='precomputed').fit(X)
    et = time.time()
    tree_features[f"feature_mds_{metric}_time_{mds_n_components}_{suffix}"] = et-st
    tree_features[f"feature_mds_{metric}_stress_{mds_n_components}_{suffix}"] = mds.stress_
    #mds_normalized = MDS(random_state=0, n_components=mds_n_components, metric=True, dissimilarity='precomputed', normalized_stress = True).fit(X)
    #tree_features[f"feature_mds_{metric}_nornalized_stress_{mds_n_components}_{suffix}"] = mds_normalized.stress_

    #for i in range(mds_n_components):
    #    tree_features[f"feature_mds_{metric}_{i}_{mds_n_components}_{suffix}"] = list(X_transform[:len(starting_trees), i])
    #st = time.time()
    #pca = PCA(n_components=pca_n_components)
    #et = time.time()
    #tree_features[f"feature_mds_{metric}_pca_time_{mds_n_components}_{suffix}"] = et - st
    #pca_transformed_x = pca.fit_transform(X_transform)
    #for i in range(pca_n_components):
    #    tree_features[f"feature_mds_{metric}_pca_{i}_{mds_n_components}_{suffix}"] = list(pca_transformed_x[:len(starting_trees), i])






def enrich_with_MDS_features(curr_run_directory, tree_features, pars_neighbors, suffix): #starting_trees = Basic_tree_features['starting_tree_object']
    overall_trees_path = os.path.join(curr_run_directory, 'overall_trees')
    pars_starting_trees = list(tree_features.loc[tree_features.starting_tree_type=='pars']["starting_tree_object"])
    overall_trees = pars_starting_trees + pars_neighbors
    unify_text_files(overall_trees, overall_trees_path, str_given=True)

    distances = np.array(RF_distances(curr_run_directory, trees_path_a=overall_trees_path, trees_path_b=None,
                                      name="RF"))
    X = np.zeros((len(overall_trees), len(overall_trees)))
    triu = np.triu_indices(len(overall_trees), 1)
    X[triu] = distances
    X = X.T
    X[triu] = X.T[triu]
    mds_and_pca(tree_features, X, overall_trees, mds_n_components = 3, metric = True, pca_n_components = 3, suffix = suffix)
    mds_and_pca(tree_features, X, overall_trees, mds_n_components = 3, metric = False, pca_n_components = 3,suffix = suffix)
    mds_and_pca(tree_features, X, overall_trees, mds_n_components = 10, metric = True, pca_n_components = 10,suffix = suffix)
    mds_and_pca(tree_features, X, overall_trees, mds_n_components = 10, metric = False, pca_n_components = 10,suffix = suffix)
    mds_and_pca(tree_features, X, overall_trees, mds_n_components = 30, metric = True, pca_n_components = 10,suffix = suffix)
    mds_and_pca(tree_features, X, overall_trees, mds_n_components = 30, metric = False, pca_n_components = 10,suffix = suffix)


def enrich_with_LLE_and_ISOMAP(all_pars_tree_distances,all_rand_tree_distances,pars_extended_tree_features_df,rand_extended_tree_features_df):

    iso = Isomap(n_components=2, n_neighbors=3).fit(all_pars_tree_distances)
    st = time.time()
    iso_pars_embedding_df = pd.DataFrame(iso.transform(all_pars_tree_distances),
                                                    columns=['feature_iso_0', 'feature_iso_1']).reset_index(drop = True)

    iso_rand_embedding_df = pd.DataFrame(iso.transform(all_rand_tree_distances),
                 columns=['feature_iso_0', 'feature_iso_1']).reset_index(drop = True)

    en = time.time()
    isomap_time = en - st
    st = time.time()
    LLE_embedding = LocallyLinearEmbedding(n_components=2,n_neighbors=3).fit(all_pars_tree_distances)
    LLE_pars_embedding_df = pd.DataFrame(LLE_embedding.transform(all_pars_tree_distances),
                                                    columns=['feature_lle_0', 'feature_lle_1']).reset_index(drop = True)

    LLE_rand_embedding_df = pd.DataFrame(LLE_embedding.transform(all_rand_tree_distances),
                 columns=['feature_lle_0', 'feature_lle_1']).reset_index(drop = True)
    en = time.time()
    lle_time = st-en

    pars_df = pd.concat([pars_extended_tree_features_df,iso_pars_embedding_df,LLE_pars_embedding_df], axis=1)
    rand_df = pd.concat([rand_extended_tree_features_df,iso_rand_embedding_df,LLE_rand_embedding_df], axis=1)

    full_df = pd.concat([pars_df,rand_df])

    try:
        full_df["feature_iso_eigen1"] = iso.kernel_pca_.eigenvalues_[0]
        full_df["feature_iso_eigen2"] = iso.kernel_pca_.eigenvalues_[1]
    except:
        full_df["feature_iso_eigen1"] = -1
        full_df["feature_iso_eigen2"] = -1
        logging.info("Problem with eigenvalues")
    full_df["feature_isomap_time"] = isomap_time
    full_df["feature_lle_time"] = lle_time
    full_df["feature_lle_reconstruction_error"] = LLE_embedding.reconstruction_error_
    return full_df


def get_basic_tree_features(msa_raw_data, curr_run_directory,msa_path,tmp_folder, args):
    basic_tree_features = msa_raw_data[
        ["starting_tree_ind", "starting_tree_type", "starting_tree_object",
         "starting_tree_ll"]].drop_duplicates().reset_index().sort_values(["starting_tree_type", "starting_tree_ind"])

    all_starting_trees_path = os.path.join(curr_run_directory, 'all_trees')
    unify_text_files(basic_tree_features['starting_tree_object'], all_starting_trees_path, str_given=True)

    optimized_tree_object_ll, optimized_tree_object_alpha, optimized_trees_file = raxml_optimize_trees_for_given_msa(
        get_local_path(msa_path), "trees_eval", all_starting_trees_path,
        tmp_folder, args.msa_type, opt_brlen=True
    )
    basic_tree_features["feature_tree_optimized_ll"] = optimized_tree_object_ll
    basic_tree_features["feature_tree_optimized_alpha"] = optimized_tree_object_alpha
    basic_tree_features["optimized_tree_object"] = generate_multiple_tree_object_from_newick_file(optimized_trees_file)
    #basic_tree_features["final_tree_objects"] = generate_multiple_tree_object_from_newick_list(basic_tree_features["final_tree_topology"])

    all_parsimony_trees_LL = basic_tree_features[basic_tree_features["starting_tree_type"] == "pars"][
        "feature_tree_optimized_ll"].drop_duplicates().tolist()
    all_random_trees_LL = basic_tree_features[basic_tree_features["starting_tree_type"] == "rand"][
        "feature_tree_optimized_ll"].drop_duplicates().tolist()
    basic_tree_features["feature_parsimony_vs_random_diff"] = np.mean(all_parsimony_trees_LL) - np.mean(
        all_random_trees_LL)
    return basic_tree_features



def tree_embeddings_pipeline(extended_tree_features_df,curr_run_directory, all_neighbors):
    all_tree_distances = np.array(list(extended_tree_features_df["tree_distances"])).reshape(
        len(extended_tree_features_df.index), -1)
    pars_extended_tree_features_df = extended_tree_features_df.loc[
        extended_tree_features_df.starting_tree_type == "pars"].copy().reset_index(drop = True)
    rand_extended_tree_features_df = extended_tree_features_df.loc[
        extended_tree_features_df.starting_tree_type == "rand"].copy().reset_index(drop = True)

    all_pars_tree_distances = np.array(list(pars_extended_tree_features_df["tree_distances"])).reshape(
        len(pars_extended_tree_features_df.index), -1)
    all_rand_tree_distances = np.array(list(rand_extended_tree_features_df["tree_distances"])).reshape(
        len(rand_extended_tree_features_df.index), -1)

    st = time.time()
    logging.info("Performing first PCA")
    n_PCA_components = 10
    pars_pca = PCA(n_components=n_PCA_components).fit(all_pars_tree_distances)  # reduce dimensions to those of parsimony trees
    if True:
        all_pars_tree_distances = pars_pca.transform(all_pars_tree_distances)
        all_rand_tree_distances = pars_pca.transform(all_rand_tree_distances)
        all_tree_distances = pars_pca.transform(all_tree_distances)
    all_tree_distances_PCA = pars_pca.transform(all_tree_distances)

    en  = time.time()
    PCA_time = en-st
    extended_tree_features_df["feature_explained_var"] = np.sum(pars_pca.explained_variance_ratio_)
    #var_explained = ""

    logging.info("Perofrming LLE and ISOMAP")
    extended_tree_features_df = enrich_with_LLE_and_ISOMAP(all_pars_tree_distances, all_rand_tree_distances,
                                                           pars_extended_tree_features_df,
                                                           rand_extended_tree_features_df)

    extended_tree_features_df["feature_PCA_time"] = PCA_time

    for i in range(n_PCA_components):
        extended_tree_features_df[f"feature_PCA_{i}"] = all_tree_distances_PCA[:,i]

    logging.info("Perofrming TSNE")
    st = time.time()
    TSNE_model = TSNE(n_components=2, init='pca', perplexity=3)
    TSNE_embedded = TSNE_model.fit_transform(all_tree_distances)
    en = time.time()
    TSNE_embedded_df = pd.DataFrame(TSNE_embedded, columns=['feature_TSNE_0', 'feature_TSNE_1']).reset_index(drop = True)
    extended_tree_features_df.reset_index(inplace=True, drop=True)
    extended_tree_features_df = pd.concat([extended_tree_features_df, TSNE_embedded_df], axis=1)
    extended_tree_features_df["feature_TSNE_time"] = en - st
    extended_tree_features_df["feature_TSNE_kl_divergence_"] = TSNE_model.kl_divergence_

    logging.info("Perofrming MDS")
    enrich_with_MDS_features(curr_run_directory, extended_tree_features_df,
                             [], suffix='only_base')
    return extended_tree_features_df


def tree_features_pipeline(msa_path, curr_run_directory, msa_raw_data, existing_tree_features_path, args):
    tmp_folder = os.path.join(curr_run_directory, 'tmp_working_dir_tree_features_pipeline')
    create_or_clean_dir(tmp_folder)
    if os.path.exists(existing_tree_features_path):
        existing_tree_features = pickle.load(open(existing_tree_features_path, "rb"))
        return existing_tree_features
    basic_tree_features = get_basic_tree_features(msa_raw_data, curr_run_directory,msa_path,tmp_folder, args)
    all_parsimony_trees = basic_tree_features[basic_tree_features["starting_tree_type"] == "pars"][
        "starting_tree_object"].drop_duplicates().tolist()
    all_parsimony_trees_LL = basic_tree_features[basic_tree_features["starting_tree_type"] == "pars"][
        "feature_tree_optimized_ll"].drop_duplicates().tolist()
    extended_tree_metrics_lst = []
    all_neighbors = []
    for index, basic_features in basic_tree_features.iterrows():
        general_tree_metrics, neighbors= single_tree_metrics(tmp_folder, all_parsimony_trees,all_parsimony_trees_LL, basic_features["optimized_tree_object"],basic_features["starting_tree_ll"], msa_path, args.msa_type, spr_iterations = args.spr_iters)
        general_tree_metrics.update(dict(basic_features))
        if basic_features["starting_tree_type"]=='pars':
            all_neighbors.extend(neighbors)
        extended_tree_metrics_lst.append(general_tree_metrics)
    extended_tree_features_df = pd.DataFrame(extended_tree_metrics_lst)
    extended_tree_features_df = tree_embeddings_pipeline(extended_tree_features_df,curr_run_directory, all_neighbors)
    tree_features = extended_tree_features_df[[col for col in extended_tree_features_df.columns if col.startswith('feature')]+["starting_tree_ind", "starting_tree_type"]]
    pickle.dump(tree_features, open(existing_tree_features_path, 'wb'))
    return tree_features


def process_all_msa_RAxML_runs(curr_run_directory, processed_dataset_path, msa_data, cpus_per_job,args,
                               perform_topology_tests=False):
    '''

    :param curr_run_directory:
    :param processed_msa_data:
    :return:
    '''
    if os.path.exists(processed_dataset_path):
        logging.info("## Using existing data features for this MSA")
        msa_data = pickle.load(open(processed_dataset_path, "rb"))
    else:
        logging.info("## Calculating features from beggining for this MSA")
        best_msa_ll = max(msa_data["final_ll"])
        msa_data["best_msa_ll"] = best_msa_ll
        best_msa_tree_topology = max(msa_data[msa_data["final_ll"] == best_msa_ll]['final_tree_topology'])
        msa_path = max(msa_data[msa_data["final_ll"] == best_msa_ll]['msa_path'])
        msa_data["rf_from_overall_msa_best_topology"] = msa_data["final_tree_topology"].apply(
            lambda x: rf_distance(curr_run_directory, x, best_msa_tree_topology, name="MSA_enrichment_RF_calculations"))
        msa_data = msa_data.sort_values(["starting_tree_type", "starting_tree_ind", "spr_radius", "spr_cutoff"])
        msa_data["final_trees_inds"] = list(range(len(msa_data.index)))
        msa_data["final_trees_inds"] = list(range(len(msa_data.index)))
        unique_trees_mapping = get_unique_trees_mapping(curr_run_directory, list(msa_data["final_tree_topology"]))
        msa_data["tree_clusters_ind"] = msa_data["final_trees_inds"].apply(lambda x: unique_trees_mapping[x])
        msa_type = args.msa_type
        if perform_topology_tests:
            try:
                per_clusters_data = msa_data.groupby(["tree_clusters_ind"]).first().reset_index()[
                    ["tree_clusters_ind", "final_tree_topology"]]
                au_test_results = au_test(per_tree_clusters_data=per_clusters_data, ML_tree=best_msa_tree_topology,
                                          msa_path=get_local_path(msa_path), cpus_per_job=cpus_per_job,
                                          name="MSA_enrichment_TREE_TEST_calculations",
                                          curr_run_directory=curr_run_directory, msa_type=msa_type)
                msa_data = msa_data.merge(pd.DataFrame(au_test_results), on="tree_clusters_ind")
            except Exception as e:
                logging.info(f"AU couldn't be estimated for current MSA")
                logging.info(f'Error details: {str(e)}')
                return pd.DataFrame()
        msa_data["delta_ll_from_overall_msa_best_topology"] = np.where(
            (msa_data["rf_from_overall_msa_best_topology"]) > 0, best_msa_ll - msa_data["final_ll"], 0)
        pickle.dump(msa_data, open(processed_dataset_path, 'wb'))
    return msa_data


def enrich_raw_data(curr_run_directory, raw_data, cpus_per_job, perform_topology_tests, args):
    '''

    :param curr_run_directory:
    :param raw_data:
    :return:  Enrich raw_data to estimate best LL for each MSA
    '''
    enriched_datasets = []
    MSAs = raw_data["msa_path"].unique()
    logging.info(f"Number of MSAs to work on: {len(MSAs)}")
    for i, msa_path in enumerate(MSAs):
        logging.info(f"Working on MSA number {i} in : {msa_path}")
        msa_folder = os.path.join(curr_run_directory, get_msa_name(msa_path, GENERAL_MSA_DIR))
        msa_data = raw_data[raw_data["msa_path"] == msa_path].copy().reset_index()
        create_dir_if_not_exists(msa_folder)
        existing_msa_features_path = os.path.join(msa_folder, "msa_features")
        existing_tree_features_path = os.path.join(msa_folder, "tree_features")
        msa_final_dataset_path = os.path.join(msa_folder, "final_dataset")
        processed_dataset_path = os.path.join(msa_folder, "processed_dataset")
        processed_msa_data = process_all_msa_RAxML_runs(msa_folder, processed_dataset_path, msa_data, cpus_per_job,args,
                                                        perform_topology_tests=perform_topology_tests)
        if not len(processed_msa_data.index) > 0:
            logging.info("no data to process")
            continue
        st = time.time()
        tree_features = tree_features_pipeline(msa_path, msa_folder, msa_data, existing_tree_features_path, args)
        ed = time.time()
        processed_msa_data['tree_features_running_time'] = ed-st
        processed_msa_data = processed_msa_data.merge(tree_features, right_on=["starting_tree_ind", "starting_tree_type"],
                                                      left_on=["starting_tree_ind", "starting_tree_type"])

        st = time.time()
        msa_features = msa_features_pipeline(msa_path, existing_msa_features_path, args)
        ed = time.time()
        processed_msa_data['msa_features_running_time'] = ed-st
        logging.info(f"MSA features: {msa_features}")
        processed_msa_data = processed_msa_data.merge(msa_features, right_on=["msa_path"], left_on=["msa_path"])

        pickle.dump(processed_msa_data, open(msa_final_dataset_path, "wb"))
        enriched_datasets.append(processed_msa_data)
        enriched_data = pd.concat(enriched_datasets)
    logging.info(f"Number of unique MSAs in final result is {len(enriched_data['msa_path'].unique())}")
    return enriched_data


def main():
    parser = feature_job_parser()
    args = parser.parse_args()
    log_file_path = os.path.join(args.curr_job_folder, "features.log")
    logging.basicConfig(filename=log_file_path, level=logging.INFO)
    curr_job_raw_data = pd.read_csv(args.curr_job_raw_path, sep=CSV_SEP)
    raw_data_with_features = enrich_raw_data(curr_run_directory=args.existing_msas_folder, raw_data=curr_job_raw_data,
                                                     cpus_per_job=args.cpus_per_job,
                                                     perform_topology_tests=args.perform_topology_tests, args = args)

    #raw_data_with_features = pd.DataFrame()
    logging.info(f'Writing enriched data to {args.features_output_path}')
    raw_data_with_features.to_csv(args.features_output_path, sep=CSV_SEP)


if __name__ == "__main__":
    main()
