
from side_code.basic_trees_manipulation import *
from side_code.raxml import *
from side_code.MSA_manipulation import get_local_path
from ML_utils.ML_algorithms_and_hueristics import ML_model, print_model_statistics_pipeline,train_test_validation_splits
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
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from scipy.spatial import distance_matrix
from sklearn.mixture import GaussianMixture
from sklearn.cross_decomposition import CCA
from scipy import stats
from sklearn import metrics
from sklearn.svm import OneClassSVM,LinearSVC, SVC, SVR
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KernelDensity
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor



def plot_svm(clf,X,Y):
    # plot the decision function for each datapoint on the grid
    xx, yy = np.meshgrid(np.linspace(-2, 2, 500), np.linspace(-2, 2, 500))
    np.random.seed(0)
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    print(Z.max())
    print(Z.min())
    data = pd.DataFrame({'xx':xx.ravel(),'yy':yy.ravel(),'Z':Z.ravel()})
    pd.DataFrame(X).to_csv('X_svm.csv')
    pd.DataFrame(Y).to_csv('Y_svm.csv')
    data.to_csv('curr_svm_plot.csv')
    plt.imshow(
        Z,
        interpolation="nearest",
        extent=(xx.min(), xx.max(), yy.min(), yy.max()),
        aspect="auto",
        origin="lower",
        cmap=plt.cm.PuOr_r,
    )
    levels = [0]
    contours = plt.contour(xx, yy, Z, levels=levels, linewidths=3, linestyles="dashed")
    #plt.colorbar()
    plt.scatter(X[:, 0], X[:, 1], s=40, c=Y,  edgecolors="k",cmap=plt.cm.Paired),
    plt.xticks(())
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.yticks(())
    plt.axis([-3, 3, -3, 3])
    plt.show()


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



def fit_SVC(svc_model, X_transformed, best_tree, name, all_results, True_global_data):
    svm = svc_model.fit(X=X_transformed, y=best_tree)


    best_svm_scores = svm.decision_function(X_transformed)[np.array(best_tree) == True]
    not_best_svm_scores = svm.decision_function(X_transformed)[np.array(best_tree) == False]
    #print(f"True global data = {True_global_data}")
    #if True_global_data is None:
    #    all_best_svm_scores = best_svm_scores
    #else:
    #    all_best_svm_scores = np.array(list(best_svm_scores) + list(svm.decision_function((True_global_data))))

    svm_results = {
        f'{name}_mean_best_score': np.mean(best_svm_scores),
                   f'{name}_max_best_score': np.max(best_svm_scores),
                   f'{name}_min_best_score': np.min(best_svm_scores),
        f'{name}_mean_non_best_score': np.mean(not_best_svm_scores),
        f'{name}_max_non_best_score': np.max(not_best_svm_scores),
        f'{name}_min_non_best_score': np.min(not_best_svm_scores),

       #f'{name}_mean_TRUE_best_score_EXCLUDE': np.mean(all_best_svm_scores),
        #f'{name}_max_TRUE_best_score_EXCLUDE': np.max(all_best_svm_scores),
        #f'{name}_min_TRUE_best_score_EXCLUDE': np.mean(all_best_svm_scores),

                   }

    all_results.update(svm_results)
    #if LOCAL_RUN:
    #     plot_svm(svm, X_transformed, best_tree)
#
#
# def fit_gmm(all_results,X_transformed, best_tree,name):
#     gmm_not_best = GaussianMixture(n_components=1, random_state=0).fit(X_transformed[np.array(best_tree) == False, :])
#     mean_overall_ll_best_trees = np.mean(gmm_not_best.score_samples(X_transformed[np.array(best_tree) == False, :]))
#     print(mean_overall_ll_best_trees)
#
#     all_results.update({f'{name}_mean_not_best_trees_gmm_1_ll_score': mean_overall_ll_best_trees})
#     mean_overall_ll_best_trees = np.mean(gmm_not_best.score_samples(X_transformed[np.array(best_tree) == True, :]))
#     print(mean_overall_ll_best_trees)
#     all_results.update({f'{name}_mean_best_trees_gmm_1_ll_score': mean_overall_ll_best_trees})

#
def kde_esitmate(X_transformed, best_tree):

    kde_x = X_transformed[np.array(best_tree) == False, :]
    kde_best = X_transformed[np.array(best_tree) == True, :]
    bandwidth = 1/(X_transformed.shape[1]*X_transformed.var())
    #print(f"b={bandwidth}")
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(kde_x) #bandwidth=bandwidth
    log_density = kde.score_samples(kde_best)
    return np.min(log_density)



def three_d_plot(xdata, ydata, zdata, best_tree):
    ax = plt.axes(projection='3d')
    c=['red' if b else 'blue' for b in best_tree]
    ax.scatter3D(xdata, ydata, zdata, c=c, cmap='Greens')
    plt.show()



def extract_2d_shape_and_plot(X_transformed, best_tree, name, X_transformed_overall_best_tree, tree_clusters_ind, final_ll_score,True_global_ll_values):

    all_results = {}

    data = pd.DataFrame({'best_tree': best_tree,'X_transformed': X_transformed.tolist(), 'final_ll_score': final_ll_score, 'tree_clusters_ind': tree_clusters_ind})
    data = data.groupby('tree_clusters_ind').first().reset_index()


    if np.sum(best_tree)>=1 and np.sum(np.array(best_tree)==False)>=2:

        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import GridSearchCV


        #reg =  LinearRegression().fit(X=np.array(list(data['X_transformed'])), y=list(data['final_ll_score']))
        #all_results.update({f'{name}_LIN_reg':reg.score(np.array(list(data['X_transformed'])), list(data['final_ll_score']))})

        min_kde = kde_esitmate(X_transformed, best_tree)
        all_results.update({f"{name}_kde_svc":min_kde})
        #print(f"min kde = {min_kde}")
        clf = SVR()
        svr  = clf.fit(X=np.array(list(data['X_transformed'])), y=list(data['final_ll_score']))
        all_results.update({f'{name}_SVR_reg': svr.score(X=np.array(list(data['X_transformed'])), y=list(data['final_ll_score']))})
        print(all_results)
        fit_SVC(SVC(), X_transformed, best_tree, f"{name}_rbf_svc", all_results, True_global_data= X_transformed_overall_best_tree)
        fit_SVC(SVC(), np.array(final_ll_score).reshape(-1,1), best_tree, f"{name}_ll_rbf_svc", all_results,
                True_global_data= True_global_ll_values)


        print("After SVC evaluations")
        if LOCAL_RUN:
            #try:
                data = pd.DataFrame({'score1': list(X_transformed[:, 0]), 'score2': list(X_transformed[:, 1]),
                                     'best_tree': best_tree})
                data['best_tree'] = data['best_tree'].apply(lambda x: 'Sample global' if x else 'Local')
                if X_transformed_overall_best_tree is not None:
                    best_data = pd.DataFrame({'score1': list(X_transformed_overall_best_tree[:, 0]), 'score2': list(X_transformed_overall_best_tree[:, 1]),
                                         })
                    best_data['best_tree'] = 'Actual global'
                else:
                    best_data = pd.DataFrame()
                total_data = pd.concat([data, best_data])
                sns.scatterplot(data=total_data, x='score1', y='score2', hue='best_tree',s=30, alpha=0.6)
                plt.show()
            #except:
            #    print("Cannot plot 2d")
    return all_results


def generate_RF_distance_matrix_statistics_final_trees(curr_run_directory, final_trees, best_tree, prefix,ll):
    if len(final_trees)==1:
        return
    RF_distance_mat = generate_RF_distance_matrix(curr_run_directory, final_trees)
    all_results = {}
    try:
        mds_models = {f'{prefix}_mds_5':MDS(n_components=5, metric = False, dissimilarity='precomputed').fit_transform(RF_distance_mat).fit_transform(RF_distance_mat)}
        for model in mds_models:
            fit_SVC(SVC(), mds_models[ model], best_tree, f"{model}_rbf_svc", all_results, True_global_data=None)
            clf = SVR()
            svr  = clf.fit(X=mds_models[ model], y=list(ll))
            all_results.update({f'{model}_SVR_reg': svr.score(X=mds_models[ model], y=list(ll))})
    except:
        logging.error("MDS not available")
    try:
        silhouete = silhouette_score(X =RF_distance_mat, metric ='precomputed', labels = np.array(best_tree))
        all_results["feature_silhouette_score"] = silhouete
    except:
        logging.info("Could not estimate Silhouette")
    print(all_results)
    if best_tree:
        n_best_trees = np.sum(best_tree)
        distances_to_others_mat =  RF_distance_mat[np.array(best_tree) == True, :][:, np.array(best_tree) == False]
        mean_distance_to_others = np.mean(distances_to_others_mat, axis = 0)
        rf_corr = (stats.spearmanr(mean_distance_to_others,np.array(ll)[np.array(best_tree)==False]).correlation)
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
    all_results["feature_mean_RF_best_vs_all"] = np.mean(distances_to_best_trees) / np.mean(RF_distances)
    all_results.update(rf_distance_metrics)
    return all_results

#
# def generate_embedding_per_tree(final_trees, tree_clusters):
#     df = pd.DataFrame({'tree_topologies': final_trees, 'clusters': tree_clusters})
#     #df_per_cluster = df.groupby('clusters').first().reset_index()
#
#     df["embeddings"] = df["tree_topologies"].apply(
#         lambda tree: get_distances_between_leaves(generate_tree_object_from_newick(tree), topology_only=True))
#     #df = df.merge(df_per_cluster, on="clusters")
#
#     final_paired_distances = np.array(list(df["embeddings"]))
#     return final_paired_distances



def generate_embedding_per_tree2(final_trees, tree_clusters):
    df = pd.DataFrame({'tree_topologies': final_trees, 'clusters': tree_clusters})


    df["embeddings"] = df["tree_topologies"].apply(
        lambda tree: get_distances_between_leaves(generate_tree_object_from_newick(tree), topology_only=True))

    #df_per_cluster = df.groupby('clusters').first().reset_index()

    #df_per_cluster["embeddings"] = df_per_cluster["tree_topologies"].apply(
    #    lambda tree: get_distances_between_leaves(generate_tree_object_from_newick(tree), topology_only=True))
    #df = df.merge(df_per_cluster, on="clusters", how = 'left')

    final_paired_distances = np.array(list(df["embeddings"]))
    return final_paired_distances


def get_embedding_distance_metrics(final_paired_distances_transformed, all_distance_metrics, prefix, model, model_name, final_trees, best_tree):
    d_mat_final = distance_matrix(final_paired_distances_transformed, final_paired_distances_transformed)
    distances = d_mat_final[np.triu_indices(n=len(final_trees), k=1)]
    all_distance_metrics.update(
        get_summary_statistics_dict(feature_name=f"{prefix}_{model_name}_distances", values=distances))
    all_distance_metrics.update({f'{prefix}_{model_name}_n_components': model["pca"].n_components_,
                                 f'{prefix}_{model_name}_var_explained': np.sum(
                                     model['pca'].explained_variance_ratio_)})
    print(all_distance_metrics)
    distances_to_other_trees_mat = d_mat_final[np.array(best_tree) == True, :][:, np.array(best_tree) == False]
    distances_to_other_trees = list(np.ravel(distances_to_other_trees_mat))
    distances_to_other_trees_features = get_summary_statistics_dict(
        feature_name=f"{prefix}_{model_name}_best_trees_distance_to_final_trees_", values=distances_to_other_trees)

    distances_to_best_trees_mat = d_mat_final[np.array(best_tree) == True, :][:, np.array(best_tree) == True]
    distances_to_best_trees = list(np.ravel(distances_to_best_trees_mat))
    distances_to_best_trees_features = get_summary_statistics_dict(
        feature_name=f"{prefix}_{model_name}_best_trees_distance_to_best_trees_", values=distances_to_best_trees)

    all_distance_metrics.update(distances_to_other_trees_features)
    all_distance_metrics.update(distances_to_best_trees_features)

    #from sklearn.cluster import KMeans
    #kmeans = KMeans(n_clusters=1, random_state=0, n_init="auto").fit(final_paired_distances_transformed)
    #distances_from_center = kmeans.transform(final_paired_distances_transformed)
    #print(1)




def generate_embedding_distance_matrix_statistics_final_trees(final_trees, best_tree, prefix, tree_clusters, True_global_trees, True_global_tree_clusters,True_global_ll_values, final_trees_ll):
    all_distance_metrics = {}
    branch_lenth_variation = np.var(
        [np.sum(tree_branch_length_metrics(generate_tree_object_from_newick(tree))["BL_list"]) for tree in final_trees])
    all_distance_metrics[f"{prefix}_bl_variation"] = branch_lenth_variation
    if LOCAL_RUN:
        models_dict = {'PCA_0.9': Pipeline(steps=[("pca", PCA(n_components=2)),]),}
    else:
        models_dict = {'PCA_0.9': Pipeline(steps=[("pca", PCA(n_components=0.9)),])}

    print()
    for model_name in models_dict:
        print(model_name)
        model = models_dict[model_name]

        final_paired_distances = generate_embedding_per_tree2(final_trees, tree_clusters)
        total_variance = np.sum(np.var(final_paired_distances,axis=0))
        print(f"Total variance={total_variance}")
        all_distance_metrics.update({f"{prefix}_{model_name}_total_var": total_variance})
        final_paired_distances_transformed = model.fit_transform(final_paired_distances)

        get_embedding_distance_metrics(final_paired_distances_transformed, all_distance_metrics, prefix, model,
                                       model_name, final_trees, best_tree)
        #final_paired_distances_transformed/= (total_variance ** 0.5)


        if True_global_trees is not None and len(True_global_trees.index)>0:
            final_paired_distances_overall = generate_embedding_per_tree2(True_global_trees, True_global_tree_clusters)
            final_paired_distances_overall_transformed = model.transform(final_paired_distances_overall)
            True_global_ll_values= np.array(True_global_ll_values).reshape(-1, 1)
            #final_paired_distances_overall_transformed/= (total_variance ** 0.5)
        else:
            final_paired_distances_overall_transformed = None
            True_global_ll_values = None


#        print(f"Variance explained: {np.sum(model['pca'].explained_variance_ratio_)}")
        print(f"Number of best trees{np.sum(best_tree)}")




        best_tree_statistics = extract_2d_shape_and_plot(final_paired_distances_transformed,best_tree, tree_clusters_ind = tree_clusters, name =f'{prefix}_{model_name}',X_transformed_overall_best_tree = final_paired_distances_overall_transformed, final_ll_score= final_trees_ll, True_global_ll_values = True_global_ll_values )
        #best_tree_statistics_norm = extract_2d_shape_and_plot(final_paired_distances_transformed_norm, best_tree,
        #                                                 tree_clusters_ind=tree_clusters, name=f'{prefix}_{model_name}_normalized',
        #                                                 X_transformed_overall_best_tree=final_paired_distances_overall_transformed_norm,
        #                                                 final_ll_score=final_trees_ll)
        #all_distance_metrics.update(best_tree_statistics)
        all_distance_metrics.update(best_tree_statistics)


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
                [get_distances_between_leaves(generate_tree_object_from_newick(tree), topology_only=False) for tree in
                 pars])
            #pars_kpca_10_model = KernelPCA(n_components=10, kernel='rbf').fit(pars_paired_distances)
            #pars_kpca_10_metrics = dimensionality_reduction_metrics(f'{prefix_name}_kpca10', pars_kpca_10_model,pars_paired_distances,n_trees= len(pars))


            pipe = Pipeline(steps=[("pca", PCA(n_components=20))]) #("scaler", StandardScaler()),
            pars_paired_distances_transformed = pipe.fit_transform(pars_paired_distances)
            dist_mat = distance_matrix(pars_paired_distances_transformed, pars_paired_distances_transformed)
            distances = dist_mat[np.triu_indices(n=(len(pars)), k=1)]
            distance_metrics = get_summary_statistics_dict(feature_name=f'{prefix_name}_PCA', values=distances)




            embedding_msa_features = {f'{prefix_name}_var_explained': np.sum(pipe.named_steps['pca'].explained_variance_ratio_),f'{prefix_name}_var_explained5': np.sum(pipe.named_steps['pca'].explained_variance_ratio_[:5],)}
            #print(embedding_msa_features)
            embedding_msa_features.update(distance_metrics)
            embedding_msa_features.update(RF_distances_metrics)
            return embedding_msa_features



