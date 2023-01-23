from side_code.raxml import *
from side_code.basic_trees_manipulation import *
from side_code.file_handling import create_or_clean_dir, create_dir_if_not_exists
from side_code.MSA_manipulation import get_alignment_data, alignment_list_to_df, get_msa_name, \
    get_local_path
from side_code.config import *
from Feature_extraction.features_job_functions import feature_job_parser
from sklearn.manifold import MDS, Isomap, TSNE, LocallyLinearEmbedding
from sklearn.decomposition import PCA
import pandas as pd
import pickle
from pypythia.predictor import DifficultyPredictor
from pypythia.prediction import get_all_features
from pypythia.raxmlng import RAxMLNG
from pypythia.msa import MSA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np
import time

def tree_embeddings_pipeline(extended_tree_features_df,curr_run_directory):

    pars_extended_tree_features_df = extended_tree_features_df.loc[
        extended_tree_features_df.starting_tree_type == "pars"].copy().reset_index(drop = True)
    #rand_extended_tree_features_df = extended_tree_features_df.loc[
    #    extended_tree_features_df.starting_tree_type == "rand"].copy().reset_index(drop = True)

    all_pars_tree_distances = np.array(list(pars_extended_tree_features_df["tree_distances"])).reshape(
        len(pars_extended_tree_features_df.index), -1)
    #all_rand_tree_distances = np.array(list(rand_extended_tree_features_df["tree_distances"])).reshape(
    #    len(rand_extended_tree_features_df.index), -1)

    st = time.time()
    logging.info("Performing first PCA")
    n_PCA_components = 20

    pars_pca_model = Pipeline([('scaling',StandardScaler()),('pca',PCA(n_components=n_PCA_components))])
    pars_pca_model.fit(all_pars_tree_distances)
    all_tree_distances = np.array(list(extended_tree_features_df["tree_distances"])).reshape(
        len(extended_tree_features_df.index), -1)
    all_tree_distances_PCA = pars_pca_model.transform(all_tree_distances)

    en  = time.time()
    PCA_time = en-st
    extended_tree_features_df["feature_explained_var"] = np.sum(pars_pca_model['pca'].explained_variance_ratio_)
    #var_explained = ""
    # logging.info("Perofrming LLE and ISOMAP")
    # extended_tree_features_df = enrich_with_LLE_and_ISOMAP(all_pars_tree_distances, all_rand_tree_distances,
    #                                                        pars_extended_tree_features_df,
    #                                                        rand_extended_tree_features_df)
    #
    # extended_tree_features_df["feature_PCA_time"] = PCA_time

    for i in range(n_PCA_components):
        extended_tree_features_df[f"feature_PCA_{i}"] = all_tree_distances_PCA[:,i]

    extended_tree_features_df["feature_PCA_time"] = PCA_time
    logging.info("Perofrming TSNE")
    st = time.time()
    TSNE_model = TSNE(n_components=2, init='pca')
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
    perform_MDS(tree_features, X, overall_trees, mds_n_components = 3, metric = True, pca_n_components = 3, suffix = suffix)
    #perform_MDS(tree_features, X, overall_trees, mds_n_components = 3, metric = False, pca_n_components = 3, suffix = suffix)
    perform_MDS(tree_features, X, overall_trees, mds_n_components=5, metric=True, pca_n_components=3, suffix=suffix)
    #perform_MDS(tree_features, X, overall_trees, mds_n_components=5, metric=False, pca_n_components=3, suffix=suffix)
    perform_MDS(tree_features, X, overall_trees, mds_n_components = 10, metric = True, pca_n_components = 10, suffix = suffix)
    #perform_MDS(tree_features, X, overall_trees, mds_n_components = 10, metric = False, pca_n_components = 10, suffix = suffix)
    perform_MDS(tree_features, X, overall_trees, mds_n_components=15, metric=True, pca_n_components=15, suffix=suffix)
    #perform_MDS(tree_features, X, overall_trees, mds_n_components=15, metric=False, pca_n_components=15, suffix=suffix)
    perform_MDS(tree_features, X, overall_trees, mds_n_components = 30, metric = True, pca_n_components = 10, suffix = suffix)
    #perform_MDS(tree_features, X, overall_trees, mds_n_components = 30, metric = False, pca_n_components = 10, suffix = suffix)



def perform_MDS(tree_features, X, starting_trees, mds_n_components, metric, pca_n_components, suffix):
    st = time.time()
    mds = MDS(random_state=0, n_components=mds_n_components, metric=metric, dissimilarity='precomputed').fit(X)
    et = time.time()
    tree_features[f"feature_mds_{metric}_time_{mds_n_components}_{suffix}"] = et-st
    tree_features[f"feature_mds_{metric}_stress_{mds_n_components}_{suffix}"] = mds.stress_


def enrich_with_LLE_and_ISOMAP(all_pars_tree_distances,all_rand_tree_distances,pars_extended_tree_features_df,rand_extended_tree_features_df):
    n_components = 2

    iso = Isomap(n_components=n_components, n_neighbors=3).fit(all_pars_tree_distances)
    st = time.time()
    iso_pars_embedding_df = pd.DataFrame(iso.transform(all_pars_tree_distances),
                                                    columns=[f'feature_iso_{i}' for i in range(n_components)]).reset_index(drop = True)

    iso_rand_embedding_df = pd.DataFrame(iso.transform(all_rand_tree_distances),
                 columns=[f'feature_iso_{i}' for i in range(n_components)]).reset_index(drop = True)

    en = time.time()
    isomap_time = en - st
    st = time.time()
    LLE_embedding = LocallyLinearEmbedding(n_components=n_components,n_neighbors=3).fit(all_pars_tree_distances)

    LLE_pars_embedding_df = pd.DataFrame(LLE_embedding.transform(all_pars_tree_distances),
                                                    columns=[f'feature_lle_{i}' for i in range(n_components)]).reset_index(drop = True)

    LLE_rand_embedding_df = pd.DataFrame(LLE_embedding.transform(all_rand_tree_distances),
                 columns=[f'feature_lle_{i}' for i in range(n_components)]).reset_index(drop = True)
    en = time.time()
    lle_time = st-en

    pars_df = pd.concat([pars_extended_tree_features_df,iso_pars_embedding_df,LLE_pars_embedding_df], axis=1)
    rand_df = pd.concat([rand_extended_tree_features_df,iso_rand_embedding_df,LLE_rand_embedding_df], axis=1)

    full_df = pd.concat([pars_df,rand_df])

    full_df["feature_isomap_time"] = isomap_time
    full_df["feature_lle_time"] = lle_time
    full_df["feature_lle_reconstruction_error"] = LLE_embedding.reconstruction_error_
    return full_df