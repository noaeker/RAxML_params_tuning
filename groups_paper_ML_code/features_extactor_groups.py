
from side_code.basic_trees_manipulation import *
from side_code.raxml import *
from side_code.MSA_manipulation import get_local_path
from ML_utils.ML_algorithms_and_hueristics import ML_model, print_model_statistics,train_test_validation_splits
import pickle
from side_code.file_handling import create_dir_if_not_exists, create_or_clean_dir, add_csvs_content
from groups_paper_ML_code.group_side_functions import *
from sklearn.decomposition import PCA
import pandas as pd
import os
import numpy as np
import time
from sklearn.manifold import MDS, Isomap, TSNE, LocallyLinearEmbedding
import timeit
from side_code.basic_trees_manipulation import get_distances_between_leaves,generate_tree_object_from_newick
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.spatial import distance_matrix
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import KernelPCA
from sklearn.manifold import SpectralEmbedding




def pct_25(values):
    return np.percentile(values, 25)


def pct_75(values):
    return np.percentile(values, 75)

def get_summary_statistics_dict(feature_name, values, funcs={'mean': np.mean,'var': np.var,
                                                             'pct_25': pct_25, 'pct_75': pct_75,
                                                             'min': np.min, 'max': np.max,
                                                             }):
    res = {}
    if values is None:
        for func in funcs:
            res.update({f'{feature_name}_{func}': None})
    else:
        for func in funcs:
            res.update({f'{feature_name}_{func}': (funcs[func])(values)})
    return res



def IQR(values):
    return np.percentile(values, 75)-np.percentile(values, 25)

def get_rf_dist_between_cols(tree_a, tree_b, curr_run_dir):
    rf_dists=[]
    for start_tree, out_tree in zip(tree_a, tree_b):
        rf_dist = rf_distance(curr_run_dir, start_tree, out_tree, name="start_vs_final")
        logging.debug(f"RF dist = {rf_dist}")
        rf_dists.append(rf_dist)
    return rf_dists

def generate_RF_distance_matrix(curr_run_directory, overall_trees):
    overall_trees_path = os.path.join(curr_run_directory, "trees")
    unify_text_files(overall_trees, overall_trees_path, str_given=True)
    distances = np.array(RF_distances(curr_run_directory, trees_path_a=overall_trees_path, trees_path_b=None,
                                      name="RF"))
    X = np.zeros((len(overall_trees), len(overall_trees)))
    triu = np.triu_indices(len(overall_trees), 1)
    X[triu] = distances
    X = X.T
    X[triu] = X.T[triu]
    return X


def estimate_entropy(vec):
    count = dict(pd.Series(vec).value_counts())
    probs = list(map(lambda x: x / sum(count.values()), count.values()))
    entropy = sum(list(map(lambda x: -x * np.log(x), probs)))
    return entropy

def generate_RF_distance_matrix_statistics_final_trees(curr_run_directory, final_trees, prefix):
    if len(final_trees)==1:
        return get_summary_statistics_dict(feature_name=f"{prefix}",values = None)
    RF_distance_mat = generate_RF_distance_matrix(curr_run_directory, final_trees)
    RF_distances = RF_distance_mat[np.triu_indices(n=len(final_trees), k=1)]
    rf_distance_metrics = get_summary_statistics_dict(feature_name=f"{prefix}",values = RF_distances)
    return rf_distance_metrics


def generate_embedding_distance_matrix_statistics_final_trees(final_trees, models_dict,prefix):
    all_distance_metrics = {}
    for model_name in models_dict:
        model = models_dict[model_name]
        final_paired_distances = np.array([get_distances_between_leaves(generate_tree_object_from_newick(tree), topology_only=False) for tree in final_trees])
        final_paired_distances_transformed = model.transform(final_paired_distances)
        d_mat_final = distance_matrix(final_paired_distances_transformed, final_paired_distances_transformed)
        distances = d_mat_final[np.triu_indices(n=len(final_trees), k=1)]
        all_distance_metrics.update(get_summary_statistics_dict(feature_name=f"{prefix}_{model_name}_",values = distances))
    return all_distance_metrics



def dimensionality_reduction_metrics(feature_name, model,pars_paired_distances,n_trees, dist_mat = None):
    if dist_mat is None:
        pars_paired_distances_transformed = model.transform(pars_paired_distances)
        dist_mat = distance_matrix(pars_paired_distances_transformed, pars_paired_distances_transformed)
    distances = dist_mat[np.triu_indices(n=(n_trees), k=1)]
    distance_metrics = get_summary_statistics_dict(feature_name= feature_name, values=distances)
    return distance_metrics





    # var_explained_pca = np.sum(pars_pca_model['pca'].explained_variance_ratio_)




def generate_calculations_per_MSA(msa_path,curr_run_dir, n_pars_tree_sampled = 100):
        prefix_name = "feature_MSA_level_distances_"
        raxml_trash_dir = os.path.join(curr_run_dir, 'raxml_trash')
        create_or_clean_dir(raxml_trash_dir)
        #msa_n_seq = max(relevant_data.loc[relevant_data.msa_path == msa_path]["feature_msa_n_seq"])
        pars_path = generate_n_tree_topologies(n_pars_tree_sampled, get_local_path(msa_path), raxml_trash_dir,
                                               seed=1, tree_type='pars', msa_type='AA')
        with open(pars_path) as trees_path:
            newicks = trees_path.read().split("\n")
            pars = [t for t in newicks if len(t) > 0]
            d_RF = generate_RF_distance_matrix(curr_run_dir, pars)
            RF_distances =d_RF[np.triu_indices(n = len(pars),k = 1)]

            RF_distances_metrics = get_summary_statistics_dict(feature_name=f"{prefix_name}_RF_distances", values  = RF_distances)
            pars_paired_distances = np.array(
                [get_distances_between_leaves(generate_tree_object_from_newick(tree), topology_only=True) for tree in
                 pars])
            pars_kpca_10_model = KernelPCA(n_components=10, kernel='rbf').fit(pars_paired_distances)
            pars_kpca_10_metrics = dimensionality_reduction_metrics(f'{prefix_name}_kpca10', pars_kpca_10_model,pars_paired_distances,n_trees= len(pars))
            pars_pca_10_model = PCA(n_components=10).fit(pars_paired_distances)
            pars_pca_10_metrics = dimensionality_reduction_metrics(f'{prefix_name}_pca10', pars_pca_10_model,
                                                                    pars_paired_distances, n_trees=len(pars))
            pars_iso_model_5 = Isomap(n_components=5).fit(pars_paired_distances)
            pars_iso_metrics_5 = dimensionality_reduction_metrics(f'{prefix_name}_iso', pars_iso_model_5,
                                                                   pars_paired_distances, n_trees=len(pars), dist_mat= pars_iso_model_5.dist_matrix_ )
            pars_iso_model_10 = Isomap(n_components=10).fit(pars_paired_distances)
            pars_iso_metrics_10 = dimensionality_reduction_metrics(f'{prefix_name}_iso', pars_iso_model_10,
                                                                   pars_paired_distances, n_trees=len(pars), dist_mat= pars_iso_model_10.dist_matrix_ )
            #pars_spectral_model = SpectralEmbedding(n_components=5).fit(pars_paired_distances)
            #pars_spectral_metrics = dimensionality_reduction_metrics(f'{prefix_name}_spectral', pars_spectral_model,
            #                                                       pars_paired_distances, n_trees=len(pars))


            embedding_msa_models  = {
                                 f'pars_kpca_10_model': pars_kpca_10_model,
                                       f'pars_pca_10_model': pars_pca_10_model,
                'pars_iso_model_5': pars_iso_model_5,
                                        'pars_iso_model_10':  pars_iso_model_10,
                #'pars_spectral_model':pars_spectral_model
                                  }  # 'MDS_raw_100': MDS_raw_100
            embedding_msa_features = {'feature_pca_10_var_explained': np.sum(pars_pca_10_model.explained_variance_)}
            embedding_msa_features.update(pars_kpca_10_metrics)
            embedding_msa_features.update(pars_pca_10_metrics)
            embedding_msa_features.update(pars_iso_metrics_5)
            embedding_msa_features.update(pars_iso_metrics_10)
            #embedding_msa_features.update(pars_spectral_metrics)
            embedding_msa_features.update(RF_distances_metrics)
            return embedding_msa_features,embedding_msa_models



