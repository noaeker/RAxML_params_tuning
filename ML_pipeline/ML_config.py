MSA_FEATURES_LIST = ['feature_n_seq', 'feature_n_loci', 'feature_constant_sites_pct', 'feature_avg_entropy',
                      'feature_25_pct_entropy', 'feature_75_pct_entropy', 'feature_median_entropy',
                      'feature_mean_gap_positions_pct', 'feature_25_pct_gaps', 'feature_75_pct_gaps',
                      'feature_median_gaps', 'feature_gap_var', 'feature_gap_max_by_min', 'feature_min_gap_pct',
                      'feature_n_unique_sites', 'feature_frac_unique_sites', 'feature_pypythia_msa_difficulty','feature_seq_to_loci', 'feature_seq_to_unique_loci']
TREE_FEATURES_LIST = ['feature_starting_tree_bool',
                      'feature_mean_branch_length', 'feature_mean_internal_branch_length',
                      'feature_mean_leaf_branch_length', 'feature_tree_MAD', 'feature_largest_branch_length',
                      'feature_minimal_branch_length', 'feature_median_branch_length', 'feature_25_pct_branch_length',
                      'feature_75_pct_branch_length', 'feature_largest_distance_between_taxa',
                      'feature_smallest_distance_between_taxa', 'feature_25_pct_distance_between_taxa',
                      'feature_75_pct_distance_between_taxa', 'feature_mean_rf_distance', 'feature_max_rf_distance',
                      'feature_min_rf_distance', 'feature_25_pct_rf_distance', 'feature_75_pct_rf_distance',
                      'feature_median_rf_distance', 'feature_optimized_ll', 'feature_optimized_tree_object_alpha',
                      'feature_median_ll_improvement', 'feature_mean_ll_improvement', 'feature_75_pct_ll_improvement',
                      'feature_25_pct_ll_improvement', 'feature_max_ll_improvement',
                      'feature_max_ll_improvement_radius', 'feature_min_ll_improvement',
                      'feature_min_ll_improvement_radius', 'feature_diff_vs_best_tree', 'feature_brlen_opt_effect',
                      ]



AVERAGED_FEATURES = [col+"_averaged_per_entire_MSA" for col in TREE_FEATURES_LIST]

#Basic analysis
BASIC_ANALYSIS_FEATURES = TREE_FEATURES_LIST+ MSA_FEATURES_LIST+ AVERAGED_FEATURES

Y_TEST_ERROR = "is_global_max"
Y_TEST_TIME = "normalized_relative_time"

#Final analysis

FINAL_MODEL_EXTRA_FEATURES = ['feature_total_time_predicted','feature_sum_of_predicted_success_probability','n_parsimony_clusters','feature_clusters_max_dist','feature_n_trees_used','feature_n_parsimony_trees_used','feature_n_random_trees_used']

FINAL_ANALYSIS_FEATURES = MSA_FEATURES_LIST+ AVERAGED_FEATURES +FINAL_MODEL_EXTRA_FEATURES





tree_search_features = ['spr_radius', 'spr_cutoff',"starting_tree_ll"]
