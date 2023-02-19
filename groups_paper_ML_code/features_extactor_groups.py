
from side_code.basic_trees_manipulation import *
from side_code.raxml import *
from side_code.MSA_manipulation import get_local_path
from ML_utils.ML_algorithms_and_hueristics import ML_model, print_model_statistics,train_test_validation_splits
import pickle
from side_code.file_handling import create_dir_if_not_exists, create_or_clean_dir, add_csvs_content
from groups_paper_ML_code.group_side_functions import *
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score,calinski_harabasz_score,davies_bouldin_score
import pandas as pd
import os
import numpy as np
import time
from sklearn.manifold import MDS, Isomap, TSNE, LocallyLinearEmbedding
import timeit
from side_code.basic_trees_manipulation import get_distances_between_leaves,generate_tree_object_from_newick
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
import seaborn as sns
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline
from scipy.spatial import distance_matrix
from sklearn.mixture import GaussianMixture
from sklearn.cross_decomposition import CCA
from scipy import stats
from sklearn import metrics
from sklearn.svm import OneClassSVM,LinearSVC
from sklearn.neighbors import KernelDensity
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor





def pct_25(values):
    return np.percentile(values, 25)


def pct_75(values):
    return np.percentile(values, 75)

def get_summary_statistics_dict(feature_name, values, funcs={'mean': np.mean,'var': np.var,
                                                             'pct_25': pct_25, 'pct_75': pct_75,
                                                             'min': np.min, 'max': np.max, 'median': np.median,
                                                             }):
    res = {}
    if values is None or len(values)==0:
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


def extract_2d_shape_and_plot(X_transformed,d_mat_final, best_tree, name):

    data = pd.DataFrame({'score1': list(X_transformed[:, 0]), 'score2': list(X_transformed[:, 1]),
                         'best_tree': best_tree})
    all_results = {}


    if np.sum(best_tree)>=1 and np.sum(np.array(best_tree)==False)>=2:

        kde_x = X_transformed[np.array(best_tree) == False, :]
        kde_best = X_transformed[np.array(best_tree) == True, :]
        bandwidth = kde_x.shape[0] ** (-1 / (kde_x.shape[1] + 4))
        bandwith2 = (kde_x.shape[0] * (kde_x.shape[1] + 2) / 4) ** (-1 / (kde_x.shape[1] + 4))
        print(f"b={bandwidth}")
        print(f"b={bandwith2}")
        kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(kde_x)
        log_density_x = kde.score_samples(kde_x)
        log_density_best = kde.score_samples(kde_best)
        print(log_density_x)
        print(log_density_best)
        all_results.update(get_summary_statistics_dict(feature_name = f'{name}_kde_x', values = log_density_x))
        all_results.update(get_summary_statistics_dict(feature_name=f'{name}_kde_best', values=log_density_best))

        all_results.update({f'{name}_mean_best_kde_score':np.mean(log_density_best),f'{name}_mean_x_kde_score':np.mean(log_density_x)})
        if LOCAL_RUN:
            sns.scatterplot(data=data, x='score1', y='score2', hue=best_tree,s=30, alpha=0.6)
            plt.show()
    print(all_results)
    return all_results


def generate_RF_distance_matrix_statistics_final_trees(curr_run_directory, final_trees, best_tree, prefix,ll):
    if len(final_trees)==1:
        return get_summary_statistics_dict(feature_name=f"{prefix}",values = None)
    RF_distance_mat = generate_RF_distance_matrix(curr_run_directory, final_trees)
    all_results = {}
    if best_tree:
        n_best_trees = np.sum(best_tree)
        distances_to_others_mat =  RF_distance_mat[np.array(best_tree) == True, :][:, np.array(best_tree) == False]
        mean_distance_to_others = np.mean(distances_to_others_mat, axis = 0)
        rf_corr = abs(stats.spearmanr(mean_distance_to_others,np.array(ll)[np.array(best_tree)==False]).correlation)
        all_results.update({f'{prefix}_corr_rf_from_best_trees_to_final_trees': rf_corr})
        distances_to_other_trees = list(np.ravel(distances_to_others_mat))
        distances_to_best_trees_mat = RF_distance_mat[np.array(best_tree) == True, :][:, np.array(best_tree) == True]
        distances_to_best_trees = distances_to_best_trees_mat[np.triu_indices(n = n_best_trees,k = 1)]
        distances_to_other_trees_features = get_summary_statistics_dict(feature_name=f"{prefix}_best_trees_rf_to_final_trees_", values=distances_to_other_trees)
        distances_to_best_trees_features = get_summary_statistics_dict(
            feature_name=f"{prefix}_best_trees_rf_to_best_trees_", values=distances_to_best_trees)
        all_results.update(distances_to_other_trees_features)
        all_results.update(distances_to_best_trees_features)
    RF_distances = RF_distance_mat[np.triu_indices(n=len(final_trees), k=1)]
    rf_distance_metrics = get_summary_statistics_dict(feature_name=f"{prefix}_rf_distances",values = RF_distances)
    all_results.update(rf_distance_metrics)
    return all_results


def generate_embedding_distance_matrix_statistics_final_trees(final_trees,best_tree, prefix,ll):
    all_distance_metrics = {}
    branch_lenth_variation = np.var(
        [np.sum(tree_branch_length_metrics(generate_tree_object_from_newick(tree))["BL_list"]) for tree in final_trees])
    all_distance_metrics[f"{prefix}_bl_variation"] = branch_lenth_variation
    models_dict = {'PCA': Pipeline(steps=[("pca", PCA(n_components=3, whiten= True))]),'PCA_whitened': Pipeline(steps=[("pca", PCA(n_components=3, whiten= True))])}
    for model_name in models_dict:
        print(model_name)
        model = models_dict[model_name]
        final_paired_distances = np.array([get_distances_between_leaves(generate_tree_object_from_newick(tree), topology_only=False) for tree in final_trees])
        final_paired_distances_transformed = model.fit_transform(final_paired_distances)
        print(f"Variance explained: {np.sum(model['pca'].explained_variance_ratio_)}")
        print(f"Number of best trees{np.sum(best_tree)}")

        d_mat_final = distance_matrix(final_paired_distances_transformed, final_paired_distances_transformed)

        best_tree_statistics = extract_2d_shape_and_plot(final_paired_distances_transformed,d_mat_final, best_tree, name =f'{prefix}_{model_name}')
        distances = d_mat_final[np.triu_indices(n=len(final_trees), k=1)]
        all_distance_metrics.update(get_summary_statistics_dict(feature_name=f"{prefix}_{model_name}_distances",values = distances))
        all_distance_metrics.update(best_tree_statistics)
        #print("model reconstruction error", model.reconstruction_error_ )
        all_distance_metrics.update({f'{prefix}_{model_name}_var_explained':np.sum(model['pca'].explained_variance_ratio_),f'{prefix}_{model_name}_var_explained1':np.sum(model['pca'].explained_variance_ratio_[:1]) })
        distances_to_other_trees_mat = d_mat_final[np.array(best_tree) == True, :][:, np.array(best_tree) == False]
        distances_to_other_trees = list(np.ravel(distances_to_other_trees_mat))
        distances_to_other_trees_features = get_summary_statistics_dict(
            feature_name=f"{prefix}_{model_name}_best_trees_distance_to_final_trees_", values=distances_to_other_trees)

        distances_to_best_trees_mat = d_mat_final[np.array(best_tree) == True, :][:, np.array(best_tree) == True]
        distances_to_best_trees = list(np.ravel(distances_to_best_trees_mat))
        distances_to_best_trees_features = get_summary_statistics_dict(
            feature_name=f"{prefix}_{model_name}_best_trees_distance_to_best_trees_", values=distances_to_best_trees)

        all_distance_metrics.update(distances_to_other_trees_features)
        all_distance_metrics.update(best_tree_statistics)
        all_distance_metrics.update(distances_to_best_trees_features)
        #all_distance_metrics.update({f'{prefix}_{model_name}_LLE_error': model.reconstruction_error_ })
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
        prefix_name = "feature_MSA_level_"
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
            #pars_kpca_10_model = KernelPCA(n_components=10, kernel='rbf').fit(pars_paired_distances)
            #pars_kpca_10_metrics = dimensionality_reduction_metrics(f'{prefix_name}_kpca10', pars_kpca_10_model,pars_paired_distances,n_trees= len(pars))

            pipe = Pipeline(steps=[("pca", PCA(n_components=20))]) #("scaler", StandardScaler()),
            pars_paired_distances_transformed = pipe.fit_transform(pars_paired_distances)
            dist_mat = distance_matrix(pars_paired_distances_transformed, pars_paired_distances_transformed)
            distances = dist_mat[np.triu_indices(n=(len(pars)), k=1)]
            distance_metrics = get_summary_statistics_dict(feature_name=f'{prefix_name}_PCA', values=distances)




            embedding_msa_features = {f'{prefix_name}_var_explained': np.sum(pipe.named_steps['pca'].explained_variance_ratio_),f'{prefix_name}_var_explained5': np.sum(pipe.named_steps['pca'].explained_variance_ratio_[:5],)}
            print(embedding_msa_features)
            embedding_msa_features.update(distance_metrics)
            embedding_msa_features.update(RF_distances_metrics)
            return embedding_msa_features



