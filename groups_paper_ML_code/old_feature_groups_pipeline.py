# print( np.sum(best_tree),len(best_tree))
# siluhette = silhouette_score(X_transformed, best_tree)
# siluhette_scores = silhouette_samples(X_transformed, best_tree)[np.array(best_tree) == True]
# best_to_outer_distance = d_mat_final[np.array(best_tree) == True,:][:,np.array(best_tree) == False].mean()
# min_best_to_outer_distance = d_mat_final[np.array(best_tree) == True,:][:,np.array(best_tree) == False].min()
# outer_to_outer_distance = d_mat_final[np.array(best_tree) == False, :][:,np.array(best_tree) == False].mean()
# my_siluhette = best_to_outer_distance/outer_to_outer_distance
# print(f"My siluhette {my_siluhette}")
# my_siluhette_min = min_best_to_outer_distance / outer_to_outer_distance
# print(f"My siluhette min {my_siluhette_min}")
# ch_score = calinski_harabasz_score(X_transformed, best_tree)
# db_score = davies_bouldin_score(X_transformed, best_tree)
# print(f"Silhouette score {siluhette}")
# print(f"CH score {ch_score}")
# print(f"DB score {db_score}")
# print(f"Silhouette score best: {np.mean(siluhette_scores)}")

# all_results.update({f'{name}_mean_best_Silhouette_score':np.mean(siluhette_scores),f'{name}_max_best_Silhouette_score':np.max(siluhette_scores),f'{name}_overall_Silhouette_score':siluhette,f'{name}_overall_ch_score':ch_score,f'{name}_overall_db_score':db_score  })

# clf = LocalOutlierFactor(n_neighbors=3,novelty=True).fit(X_transformed[np.array(best_tree)==False,:])
# d = clf.decision_function(X_transformed[np.array(best_tree)==True,:])
# print(f"d={d}")


# data['score1_normalized'] = abs((RobustScaler().fit_transform(data[['score1']])))
#
#
# data['score2_normalized'] = abs((RobustScaler().fit_transform(data[['score2']])))
#
# data['dist_from_center_score1'] = abs(data['score1'] - data['score1'].mean())
# data['dist_from_center_score2'] = abs(data['score2'] - data['score2'].mean())
#
# best_tree_statistics = data.loc[data.best_tree == 1]
# all_results = {}
# dist_from_center1_dict = get_summary_statistics_dict(f'{name}_final_trees_center_1',best_tree_statistics['score1_normalized'], funcs={'mean':np.mean,'max':np.max})
# dist_from_center2_dict = get_summary_statistics_dict(f'{name}_final_trees_center_2',
#                                                      best_tree_statistics['score2_normalized'],
#                                                      funcs={'mean': np.mean, 'max': np.max})
# all_results.update(dist_from_center1_dict)
# all_results.update(dist_from_center2_dict)

# gmm_not_best = GaussianMixture(n_components=1, random_state=0).fit(X_transformed[np.array(best_tree)==False,:])
# mean_overall_ll_best_trees = np.mean(gmm_not_best.score_samples(X_transformed[np.array(best_tree)==False,:]))
# print(mean_overall_ll_best_trees)

# all_results.update({f'{name}_mean_not_best_trees_gmm_1_ll_score': mean_overall_ll_best_trees})
# mean_overall_ll_best_trees =np.mean(gmm_not_best.score_samples(X_transformed[np.array(best_tree)==True,:]))
# print(mean_overall_ll_best_trees)
# all_results.update({f'{name}_mean_best_trees_gmm_1_ll_score': mean_overall_ll_best_trees})


# cca = CCA(n_components=1).fit([[l] for l in ll],final_paired_distances_transformed)
from sklearn.preprocessing import MinMaxScaler
# cca_y = cca.y_scores_.flatten()
# cca_y_transgormed = (cca_y-cca_y.min())/ (cca_y.max()-cca_y.min())
# cca_y_best = np.mean(cca_y_transgormed[np.array(best_tree)==True])
# print(f"CCA {cca_y_best}")


# kde_x = X_transformed[np.array(best_tree) == False, :]
# kde_best = X_transformed[np.array(best_tree) == True, :]
# bandwidth = kde_x.shape[0] ** (-1 / (kde_x.shape[1] + 4))
# bandwith2 = (kde_x.shape[0] * (kde_x.shape[1] + 2) / 4) ** (-1 / (kde_x.shape[1] + 4))
# print(f"b={bandwidth}")
# print(f"b={bandwith2}")
# kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(kde_x)
# log_density_x = kde.score_samples(kde_x)
# log_density_best = kde.score_samples(kde_best)
# print(log_density_x)
# print(log_density_best)
# all_results.update(get_summary_statistics_dict(feature_name = f'{name}_kde_x', values = log_density_x))
# all_results.update(get_summary_statistics_dict(feature_name=f'{name}_kde_best', values=log_density_best))

# all_results.update({f'{name}_mean_best_kde_score':np.mean(log_density_best),f'{name}_mean_x_kde_score':np.mean(log_density_x)})
