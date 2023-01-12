import sys

if sys.platform == "linux" or sys.platform == "linux2":
    PROJECT_ROOT_DIRECRTORY = "/groups/pupko/noaeker/RAxML_params_tuning"
else:
    PROJECT_ROOT_DIRECRTORY = "/Users/noa/Workspace/RAxML_params_tuning"
sys.path.append(PROJECT_ROOT_DIRECRTORY)

from side_code.SPR_moves import *
from scipy.stats import skew, kurtosis
from sklearn.cluster import AgglomerativeClustering
import networkx as nx
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
from side_code.config import *
from side_code.file_handling import create_dir_if_not_exists
from ML_pipeline.side_functions import get_ML_parser
from ML_pipeline.ML_pipeline_procedures import get_average_results_on_default_configurations_per_msa,edit_raw_data_for_ML
from ML_pipeline.ML_algorithms_and_hueristics import ML_model, print_model_statistics, train_test_validation_splits,variable_importance
import pandas as pd
from sklearn.cluster import DBSCAN
import lightgbm
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
import os
import pickle
import numpy as np
import argparse
from ML_pipeline.ML_algorithms_and_hueristics import train_test_validation_splits,variable_importance, model_metrics
from sklearn.metrics import matthews_corrcoef, roc_auc_score, average_precision_score, accuracy_score, precision_score, \
    recall_score, PrecisionRecallDisplay



def pct_25(values):
    return np.percentile(values, 25)


def pct_75(values):
    return np.percentile(values, 75)


def IQR(values):
    return np.percentile(values, 75)-np.percentile(values, 25)

def get_rf_dist_between_cols(tree_a, tree_b, curr_run_dir):
    rf_dists=[]
    for start_tree, out_tree in zip(tree_a, tree_b):
        rf_dists.append (rf_distance(curr_run_dir, start_tree, out_tree, name="start_vs_final"))
    return rf_dists

def generate_distance_matrix(curr_run_directory, overall_trees):
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

def perform_MDS(curr_run_directory,overall_trees, n_seq):
    #distance_mat_norm = generate_distance_matrix(curr_run_directory,overall_trees)/(2*n_seq-6)
    distance_mat_raw = generate_distance_matrix(curr_run_directory, overall_trees)
    #mds_norm = MDS(random_state=0, n_components=3, metric=True, dissimilarity='precomputed').fit(distance_mat_norm)
    mds_raw = MDS(random_state=0, n_components=10, metric=True, dissimilarity='precomputed').fit(distance_mat_raw)
    return pd.Series([mds_raw.stress_,np.mean(distance_mat_raw)])


def estimate_entropy(vec):
    count = dict(pd.Series(vec).value_counts())
    probs = list(map(lambda x: x / sum(count.values()), count.values()))
    entropy = sum(list(map(lambda x: -x * np.log(x), probs)))
    return entropy

def generate_distance_matrix_statistics(curr_run_directory,df, col_output,msa_res):
    n_seq = np.max(df["feature_msa_n_seq"])
    best_tree_ind = df["is_best_tree"].reset_index().idxmax()[0]
    distance_mat_opt = generate_distance_matrix(curr_run_directory, df[col_output])/(2*n_seq-6)
    clustering = DBSCAN(eps=0.1, min_samples=2, metric= 'precomputed').fit(distance_mat_opt)
    best_tree_outlier = clustering.labels_[best_tree_ind]==-1
    best_tree_label = clustering.labels_[best_tree_ind]
    best_tree_label_frac = np.mean(clustering.labels_ == best_tree_label)
    pct_non_outliers =  np.mean(clustering.labels_>-1)
    number_of_clusters = len(np.unique(clustering.labels_))
    return pd.Series([best_tree_outlier, pct_non_outliers,number_of_clusters,best_tree_label_frac ])


def get_mean_rf_distance(curr_run_directory,df, col):
    overall_trees = df[col]
    distance_mat = generate_distance_matrix(curr_run_directory,overall_trees)
    return pd.Series([np.mean(distance_mat), np.var(distance_mat), np.min(distance_mat), np.max(distance_mat), pct_25(distance_mat),pct_75(distance_mat)])



def get_summary_statistics_dict(feature_name, values, funcs={'mean': np.mean,'median': np.mean,'var': np.var,
                                                             'pct_25': pct_25, 'pct_75': pct_75,
                                                             'min': np.min, 'max': np.max,
                                                             }):
    res = {}
    for func in funcs:
        res.update({f'{feature_name}_{func}': (funcs[func])(values)})
    return res

def get_file_paths(args):
    new_folder = os.path.join(args.baseline_folder, args.name)
    create_dir_if_not_exists(new_folder)
    return {"features_path": f"{args.baseline_folder}/all_features{CSV_SUFFIX}",
            }



def get_average_results_on_default_configurations_per_msa(curr_run_dir,default_data, n_sample_points, seed, n_pars, n_rand, MSA_res
                                                          ):
    msa_level_cols = ["msa_path", "feature_msa_n_seq", "feature_msa_n_loci", "feature_msa_pypythia_msa_difficulty"]
    tree_level_cols = ["starting_tree_type","starting_tree_object","final_tree_topology","delta_ll_from_overall_msa_best_topology","is_global_max", "feature_tree_optimized_ll","final_ll"]
    default_data = default_data[msa_level_cols+tree_level_cols]
    logging.info(f"Number of MSAs in default data is {len(default_data['msa_path'].unique())}")
    default_results = pd.DataFrame()
    for i in range(n_sample_points):
        logging.info(f"i = {i}/{n_sample_points}")
        seed = seed + 1
        sampled_data_parsimony = default_data[default_data["starting_tree_type"] == "pars"].groupby(
            by=msa_level_cols).sample(n=n_pars, random_state=seed)
        sampled_data_random = default_data[default_data["starting_tree_type"] == "rand"].groupby(
            by=msa_level_cols).sample(n=n_rand, random_state=seed)
        sampled_data = pd.concat([sampled_data_random, sampled_data_parsimony])

        sampled_data['best_sample_ll'] = sampled_data.groupby('msa_path')['final_ll'].transform(max)
        #sampled_data["best_sampled_tree_topology"] = sampled_data.loc[sampled_data.final_ll==sampled_data.best_sample_ll].groupby('msa_path')['final_tree_topology'].transform(max)
        #sampled_data["rf_from_sampled_msa_best_topology"] = sampled_data.apply(
        #    lambda x: rf_distance(curr_run_dir, sampled_data.best_sampled_tree_topology, sampled_data.final_tree_topology, name="MSA_enrichment_RF_calculations"),axis=1)
        sampled_data["is_best_tree"] = sampled_data["final_ll"]>=sampled_data['best_sample_ll']-0.1


        sampled_data["log_likelihood_diff"] = sampled_data["final_ll"] - sampled_data[
            "feature_tree_optimized_ll"]
        sampled_data["start_vs_end"] = get_rf_dist_between_cols(sampled_data["starting_tree_object"],sampled_data["final_tree_topology"], curr_run_dir)
        sampled_data["normalized_final_ll"] = sampled_data.groupby('msa_path')["final_ll"].transform(lambda x: (x-x.mean()/x.std()))
        #sampled_data["final_o"] = sampled_data["final_tree_topology"].apply(lambda x:Tree(x, format=1))
        distance_matrix_summary_statistics = sampled_data.groupby('msa_path').apply(lambda x:generate_distance_matrix_statistics(curr_run_dir,x, col_output = 'final_tree_topology',msa_res = MSA_res)).reset_index()

        distance_matrix_summary_statistics.columns = ["msa_path","feature_DBSCAN_best_tree_outlier", "feature_DBSCAN_pct_non_outliers","feature_DBSCAN_number_of_clusters","feature_DBSCAN_pct_of_best_tree_cluster"]
        general_run_metrics = sampled_data.groupby(
            by=msa_level_cols).agg(feature_pct_best = ('is_best_tree', np.mean) ,default_final_err = ('delta_ll_from_overall_msa_best_topology', np.min),
                                                default_status = ('is_global_max',np.max),
                                                feature_final_ll_var = ('final_ll', np.var),
                                                feature_final_ll_skew=('final_ll', skew),
                                                feature_final_ll_kutosis=('final_ll', kurtosis),
                                                feature_max_ll_std = ('normalized_final_ll', np.max)
                                                ).reset_index()
        mds_per_final_tree = sampled_data.groupby('msa_path').apply(lambda df: perform_MDS(curr_run_dir,df['final_tree_topology'], max(df['feature_msa_n_seq']))).reset_index()
        mds_per_final_tree.columns = ['msa_path','feature_mds_rf_dist_final_trees_raw','feature_mean_rf_dist_final_trees_raw']

        mean_rf_per_final_tree = sampled_data.groupby('msa_path').apply(lambda df: get_mean_rf_distance(curr_run_dir,df,col='final_tree_topology')).reset_index()
        mean_rf_per_final_tree.columns = ['msa_path','feature_mean_rf_final_trees','feature_var_rf_final_trees','feature_min_rf_final_trees','feature_max_rf_final_trees','feature_25_rf_final_trees','feature_75_rf_final_trees']
        mean_rf_per_pars_starting_tree = sampled_data.loc[sampled_data.starting_tree_type=='pars'].groupby('msa_path').apply(lambda df: get_mean_rf_distance(curr_run_dir,df,col='starting_tree_object')).reset_index()
        mean_rf_per_pars_starting_tree.columns = ['msa_path','feature_mean_rf_pars_trees','feature_var_rf_pars_trees','feature_min_rf_pars_trees','feature_max_rf_pars_trees','feature_25_rf_pars_trees','feature_75_rf_pars_trees']

        pars_run_metrics = sampled_data.loc[sampled_data.starting_tree_type=='pars'].groupby('msa_path').agg(feature_mean_pars_ll_diff = ('log_likelihood_diff', np.mean),feature_var_pars_ll_diff = ('log_likelihood_diff', np.var), feature_mean_pars_rf_diff = ('start_vs_end', np.mean), feature_var_pars_vs_final_rf_diff = ('start_vs_end', np.var),feature_min_pars_vs_final_rf_diff = ('start_vs_end', np.min),feature_max_pars_vs_final_rf_diff = ('start_vs_end', np.max),  feature_pars_ll_skew=('feature_tree_optimized_ll', skew),
                                                feature_pars_ll_kutosis=('feature_tree_optimized_ll', kurtosis), feature_mean_pars_global_max = ('is_best_tree', np.mean))

        pars_final_corr = sampled_data.groupby('msa_path')[['final_ll','feature_tree_optimized_ll']].corr().unstack().iloc[:,1].reset_index()
        pars_final_corr.columns = ['msa_path','feature_corr_pars_final']
        rand_run_metrics = sampled_data.loc[sampled_data.starting_tree_type == 'rand'].groupby('msa_path').agg(
            feature_mean_rand_ll_diff=('log_likelihood_diff', np.mean),
            feature_var_rand_ll_diff=('log_likelihood_diff', np.var),
            feature_mean_rand_global_max=('is_best_tree', np.mean)
            )
        general_run_metrics = general_run_metrics.merge(pars_final_corr, on = 'msa_path')
        general_run_metrics = general_run_metrics.merge(mds_per_final_tree, on = 'msa_path')
        general_run_metrics = general_run_metrics.merge(rand_run_metrics, on = 'msa_path')
        general_run_metrics = general_run_metrics.merge(pars_run_metrics, on = 'msa_path')
        general_run_metrics = general_run_metrics.merge(mean_rf_per_final_tree, on = 'msa_path')
        general_run_metrics = general_run_metrics.merge(mean_rf_per_pars_starting_tree, on = 'msa_path')
        general_run_metrics = general_run_metrics.merge(distance_matrix_summary_statistics, on = 'msa_path')
        default_results = default_results.append(general_run_metrics)
    return default_results


def generate_calculations_per_MSA(curr_run_dir, relevant_data,msa_res_path):
    msa_res = {}
    raxml_trash_dir = os.path.join(curr_run_dir, 'raxml_trash')
    create_dir_if_not_exists(raxml_trash_dir)
    for msa_path in relevant_data["msa_path"].unique():
        msa_n_seq = max(relevant_data.loc[relevant_data.msa_path == msa_path]["feature_msa_n_seq"])
        pars_path = generate_n_tree_topologies(args.n_pars_trees_sample, get_local_path(msa_path), raxml_trash_dir,
                                               seed=1, tree_type='pars', msa_type='AA')
        with open(pars_path) as trees_path:
            newicks = trees_path.read().split("\n")
            pars = [t for t in newicks if len(t) > 0]
            MDS_res = perform_MDS(curr_run_dir, pars, msa_n_seq)
            MDS_raw = MDS_res.iloc[0]
            mean_dist_raw = MDS_res.iloc[1]
            msa_res[msa_path] = {'MDS_raw': MDS_raw, 'mean_dist_raw': mean_dist_raw, 'pars_trees': pars}
            create_or_clean_dir(raxml_trash_dir)
    with open(msa_res_path, 'wb') as MSA_RES:
        pickle.dump(msa_res, MSA_RES)
    return msa_res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', action='store', type=str,
                        default="/Users/noa/Workspace/raxml_deep_learning_results/test_single_tree_data_err.tsv")
    parser.add_argument('--curr_working_dir', default =  "/Users/noa/Workspace/raxml_deep_learning_results/new_grouping_test")
    parser.add_argument('--n_pars_trees',action='store', type=int, default= 5 )
    parser.add_argument('--n_rand_trees', action='store', type=int, default=5)
    parser.add_argument('--n_iterations', action='store', type=int, default=10)
    parser.add_argument('--n_pars_trees_sample',type=int, default=50)
    parser.add_argument('--name', type=str, default="groups_run")
    parser.add_argument('--filter_on_default_data',action = 'store_true', default= False)
    parser.add_argument('--n_jobs', type=int, default= 1)

    args = parser.parse_args()
    curr_run_dir = os.path.join(args.curr_working_dir, args.name)
    create_dir_if_not_exists(curr_run_dir)
    relevant_data = pd.read_csv(args.file_path, sep = '\t')
    if args.filter_on_default_data:
        if not LOCAL_RUN:
            relevant_data = relevant_data[relevant_data["type"] == "default"]
        else:
            relevant_data = relevant_data.loc[relevant_data.equal_to_default_config]
    relevant_data["is_global_max"] = (relevant_data["delta_ll_from_overall_msa_best_topology"] <= 0.1).astype('int')
    #relevant_data = relevant_data.loc[relevant_data.feature_msa_pypythia_msa_difficulty>0.2]
    #msas = relevant_data["msa_path"].unique()[:10]
    #relevant_data = relevant_data.loc[relevant_data.msa_path.isin(msas)]
    results_path = os.path.join(curr_run_dir,'group_results.tsv')
    log_file_path = os.path.join(curr_run_dir,"log_file")
    logging.basicConfig(filename=log_file_path, level=logging.INFO)

    #msa_res = {}
    #msa_res_path = os.path.join(curr_run_dir,'msa_res')
    #if os.path.exists(msa_res_path):
    #    msa_res = pickle.load(open(msa_res_path,"rb"))
    #else:
    #    msa_res = generate_calculations_per_MSA(curr_run_dir, relevant_data,msa_res_path)
    #logging.info(f"MSA res is of len {len(msa_res)}")
    if not os.path.exists(results_path):
        logging.info("Generating results file")
        results = get_average_results_on_default_configurations_per_msa(curr_run_dir,relevant_data, n_sample_points=args.n_iterations, seed=1, n_pars =args.n_pars_trees, n_rand = args.n_rand_trees, MSA_res= msa_res)
        results["n_pars_trees"] = args.n_pars_trees
        results["n_rand_trees"] = args.n_rand_trees
        results.to_csv(results_path, sep= '\t')
    else:
        logging.info("Reading existing results file")
        results = pd.read_csv(results_path, sep = '\t', index_col= False)


    #results["feature_mds_raw_on_pars_trees"] = results["msa_path"].apply(lambda x: msa_res[x]['MDS_raw'])
    #results["feature_mean_dist_raw_on_pars_trees"] = results["msa_path"].apply(lambda x: msa_res[x]['mean_dist_raw'])
    results["feature_mean_ll_pars_vs_rand"] = results["feature_mean_pars_ll_diff"]/results["feature_mean_rand_ll_diff"]
    results["feature_var_ll_pars_vs_rand"] = results["feature_var_pars_ll_diff"] / results[
        "feature_var_rand_ll_diff"]


    #for col in results.columns:
    #    if 'mds' in col:
    #        results[col] = np.sqrt(results[col]+0.001)
    train, test, val = train_test_validation_splits(results,test_pct= 0.3, val_pct=0, msa_col_name = 'msa_path')
    #X_train = train[["feature_mean_rf_final_trees","feature_msa_n_seq","feature_msa_n_loci","feature_msa_pypythia_msa_difficulty"]]
    #X_test = test[["feature_mean_rf_final_trees","feature_msa_n_seq","feature_msa_n_loci","feature_msa_pypythia_msa_difficulty"]]
    X_train = train[[col for col in train.columns if col.startswith('feature') ]]
    X_test = test[[col for col in train.columns if col.startswith('feature') ]]#+['mean_predicted_failure']
    y_train = train["default_status"]
    y_test = test["default_status"]
    groups = train["msa_path"]
    model_path = os.path.join(curr_run_dir,'group_classification_model')
    vi_path=  os.path.join(curr_run_dir,'group_classification_vi.tsv')
    metrics_path = os.path.join(curr_run_dir, 'group_classification_metrics.tsv')
    group_metrics_path = os.path.join(curr_run_dir, 'group_classification_group_metrics.tsv')
    #final_csv_path = os.path.join(curr_run_dir,"performance_on_test.tsv")

    model = ML_model(X_train, groups, y_train, n_jobs = args.n_jobs, path = model_path, classifier=True, model='lightgbm', calibrate=True, name="", large_grid = args.large_grid, do_RFE = True, n_cv_folds = 3)
    #model = lightgbm.LGBMClassifier().fit(X_train, y_train)
    #model = LogisticRegressionCV(random_state=0).fit(X_train, y_train)
    #predicted_proba_test = model['best_model'].predict_proba((model['selector'].transform(X_test)))[:, 1]
    print_model_statistics(model, X_train, X_test, None, y_train, y_test, None, is_classification = True, vi_path=vi_path,
                           metrics_path=metrics_path,
                           group_metrics_path=group_metrics_path, name = "", sampling_frac = -1, test_MSAs = test["msa_path"], feature_importance=True)

if __name__ == "__main__":
    main()
