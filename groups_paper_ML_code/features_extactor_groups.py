
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
import timeit
from side_code.basic_trees_manipulation import get_distances_between_leaves,generate_tree_object_from_newick
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.spatial import distance_matrix
from sklearn.mixture import GaussianMixture




def pct_25(values):
    return np.percentile(values, 25)


def pct_75(values):
    return np.percentile(values, 75)

def get_summary_statistics_dict(feature_name, values, funcs={'mean': np.mean,'median': np.mean,'var': np.var,
                                                             'pct_25': pct_25, 'pct_75': pct_75,
                                                             'min': np.min, 'max': np.max,
                                                             }):
    res = {}
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

def generate_RF_distance_matrix_statistics_final_trees(curr_run_directory, final_trees):
    prefix = "final_trees_level_embedding_"
    RF_distance_mat = generate_RF_distance_matrix(curr_run_directory, final_trees)
    RF_distances = RF_distance_mat[np.triu_indices(n=len(final_trees), k=1)]
    rf_distance_metrics = get_summary_statistics_dict(feature_name=f"{prefix}_RF_",values = RF_distances)
    return rf_distance_metrics


def generate_embedding_distance_matrix_statistics_final_trees(final_trees, pca_model, gm_1, gm_3, gm_5):
    prefix = "final_trees_level_embedding_"
    final_paired_distances = np.array([get_distances_between_leaves(generate_tree_object_from_newick(tree), topology_only=False) for tree in final_trees])
    final_paired_distances_pca = pca_model.transform(final_paired_distances)
    d_final = distance_matrix(final_paired_distances_pca, final_paired_distances_pca)
    pars_distances = d_final[np.triu_indices(n = len(final_trees), k = 1)]
    pars_distances_metrics =  get_summary_statistics_dict(feature_name=f"{prefix}_embedding_new",values = pars_distances)
    pars_distances_metrics .update({f'{prefix}gm_n_labels_3': len(np.unique(gm_3.predict(final_paired_distances_pca))) ,f'{prefix}gm_n_labels_5': len(np.unique(gm_5.predict(final_paired_distances_pca))), f'{prefix}gm_final_trees_score_1':gm_1.score(final_paired_distances_pca),f'{prefix}gm_final_trees_score_3':gm_3.score(final_paired_distances_pca),f'{prefix}gm_final_trees_score_5':gm_5.score(final_paired_distances_pca) })
    return pars_distances_metrics



def generate_calculations_per_MSA(msa_path,curr_run_dir, n_pars_tree_sampled = 100):
        prefix_name = "MSA_level_embedding_"
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
            pars_paired_distances = np.array([get_distances_between_leaves(generate_tree_object_from_newick(tree), topology_only = True) for tree in pars])
            pars_pca_model = Pipeline([('scaling', StandardScaler()), ('pca', PCA(n_components=50))])
            pars_paired_distances_pca = pars_pca_model.fit_transform(pars_paired_distances)

            gm_model_1 = GaussianMixture(n_components=1, random_state=0).fit(pars_paired_distances_pca)
            gm_model_3 = GaussianMixture(n_components=3, random_state=0).fit(pars_paired_distances_pca)
            gm_model_5 = GaussianMixture(n_components=5, random_state=0).fit(pars_paired_distances_pca)

            var_explained_pca = np.sum(pars_pca_model['pca'].explained_variance_ratio_)
            d_embedding = distance_matrix(pars_paired_distances_pca, pars_paired_distances_pca)
            embedding_distances = d_embedding[np.triu_indices(n = len(pars),k = 1)]

            embedding_msa_features  = {f'{prefix_name}MDS_raw_30_embedding':MDS(random_state=0, n_components=30, metric = True, dissimilarity='precomputed').fit(d_embedding).stress_,'MDS_raw_30_RF': MDS(random_state=0, n_components=30, metric = True, dissimilarity='precomputed').fit(d_RF).stress_, 'mean_dist_embedding': np.mean(embedding_distances),
                                 f'{prefix_name}var_dist_embedding': np.var(embedding_distances), 'min_dist_embedding': np.min(embedding_distances),
                                 f'{prefix_name}max_dist_embedding': np.max(embedding_distances), 'mean_dist_RF': np.mean(RF_distances),
                                 f'{prefix_name}var_dist_RF': np.var(RF_distances), 'min_dist_RF': np.min(RF_distances),
                                 f'{prefix_name}max_dist_RF': np.max(RF_distances),
                                 f'{prefix_name}bic_gm_1':gm_model_1.bic(pars_paired_distances_pca),
                                 f'{prefix_name}bic_gm_3':gm_model_3.bic(pars_paired_distances_pca),
                                 f'{prefix_name}bic_gm_5':gm_model_5.bic(pars_paired_distances_pca),
                                 f'{prefix_name}ll_gm_1': gm_model_1.score(pars_paired_distances_pca),
                                 f'{prefix_name}ll_gm_3': gm_model_3.score(pars_paired_distances_pca),
                                 f'{prefix_name}ll_gm_5': gm_model_5.score(pars_paired_distances_pca),
                                 f'{prefix_name}var_exlained_pca': var_explained_pca,
                                 'PCA_model': pars_pca_model,
                                'gm_1_model': gm_model_1,
                                  'gm_3_model': gm_model_3,
                                  'gm_5_model' : gm_model_5,
                                  }  # 'MDS_raw_100': MDS_raw_100
            return embedding_msa_features



