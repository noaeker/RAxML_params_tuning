import sys

if sys.platform == "linux" or sys.platform == "linux2":

    PROJECT_ROOT_DIRECRTORY = "/groups/pupko/noaeker/RAxML_params_tuning"
else:
    PROJECT_ROOT_DIRECRTORY = "/Users/noa/Workspace/RAxML_params_tuning"
sys.path.append(PROJECT_ROOT_DIRECRTORY)

from side_code.raxml import *
from side_code.basic_trees_manipulation import *
from side_code.file_handling import create_or_clean_dir, create_dir_if_not_exists
from side_code.MSA_manipulation import get_alignment_data, alignment_list_to_df, get_msa_name, \
    get_local_path
from side_code.config import *
from feature_extraction.features_job_functions import feature_job_parser
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

def pct_25(values):
    return np.percentile(values, 25)


def pct_75(values):
    return np.percentile(values, 75)

def get_summary_statistics_dict(feature_name, values, funcs={'mean': np.mean, 'median': np.mean, 'var': np.var,
                                                             'pct_25': pct_25, 'pct_75': pct_75,
                                                             'min': np.min, 'max': np.max,
                                                             }):
    res = {}
    for func in funcs:
        res.update({f'{feature_name}_{func}': (funcs[func])(values)})
    return res


def pypythia(msa_path, msa_type):
    predictor = DifficultyPredictor(open(f"{PROJECT_ROOT_DIRECRTORY}/pypythia/predictor.pckl", "rb"))
    raxmlng = RAxMLNG(RAXML_NG_EXE)
    msa = MSA(msa_path)
    model = "WAG+G" if msa_type == "AA" else "GTR+G"
    msa_features = get_all_features(raxmlng, msa, model)
    difficulty = predictor.predict(msa_features)
    return difficulty


def get_msa_stats(msa_path, msa_type):
    msa_data = get_alignment_data(get_local_path(msa_path))
    n_seq, n_loci = len(msa_data), len(msa_data[0].seq)
    all_msa_features = {"feature_msa_n_seq": n_seq, "feature_msa_n_loci": n_loci, "msa_path": msa_path,
                        "msa_name": get_msa_name(msa_path, GENERAL_MSA_DIR), "feature_msa_type": msa_type}
    alignment_data = get_alignment_data(get_local_path(msa_path))
    alignment_df = alignment_list_to_df(alignment_data)
    alignment_df_fixed = alignment_df.replace('-', np.nan)
    gap_fracs_per_seq = alignment_df_fixed.isna().sum(axis=1) / n_loci
    gap_fracs_per_loci = alignment_df_fixed.isna().sum(axis=0) / n_seq
    alignment_df_unique = alignment_df.T.drop_duplicates().T
    counts_per_position = [dict(alignment_df_fixed[col].value_counts(dropna=True)) for col in list(alignment_df)]
    probabilities = [list(map(lambda x: x / sum(counts_per_position[col].values()), counts_per_position[col].values()))
                     for col in
                     list(alignment_df)]
    entropy = [sum(list(map(lambda x: -x * np.log(x), probabilities[col]))) for col in list(alignment_df)]
    constant_sites_pct = sum([1 for et in entropy if et == 0]) / len(entropy)
    msa_difficulty = pypythia(get_local_path(msa_path), msa_type=msa_type)

    all_msa_features.update({"feature_msa_constant_sites_pct": constant_sites_pct,
                             "feature_msa_n_unique_sites": len(alignment_df_unique.columns),
                             "feature_msa_pypythia_msa_difficulty": msa_difficulty})

    multi_dimensional_features = {"feature_msa_entropy": entropy, "feature_msa_gap_fracs_per_seq": gap_fracs_per_seq,
                                  "feature_msa_gap_fracs_per_loci": gap_fracs_per_loci}

    for feature in multi_dimensional_features:
        all_msa_features.update(get_summary_statistics_dict(feature, multi_dimensional_features[feature]))
    return all_msa_features








def process_all_msa_runs(curr_run_directory,msa_path, msa_data):
    '''

    :param curr_run_directory:
    :param processed_msa_data:
    :return:
    '''
    logging.info("## Calculating features from beggining for this MSA")
    msa_type = np.max(msa_data['msa_type'])
    msa_data = msa_data.loc[((msa_data.program=='IQTREE')& (msa_data.type=='default')) | ((msa_data.program=='RAxML')& (msa_data.type!='default'))]
    msa_data["final_trees_inds"] = list(range(len(msa_data.index)))
    unique_trees_mapping = get_unique_trees_mapping(curr_run_directory, list(msa_data["final_tree_topology"]))
    msa_data["tree_clusters_ind"] = msa_data["final_trees_inds"].apply(lambda x: unique_trees_mapping[x])
    if len(msa_data["file_name"].unique())>1:
        logging.info("MSA of both IQTREE and RAxML")
        trees_path = os.path.join(curr_run_directory, 'trees_for_ll_evaluation')

        unique_trees = msa_data.groupby(['tree_clusters_ind'])["final_tree_topology"].first().reset_index()


        final_trees = list(unique_trees["final_tree_topology"])
        with open(trees_path,'w') as TREES_ML:
            TREES_ML.writelines(final_trees)
        trees_ll_on_data, tree_alpha, optimized_trees_final_path = raxml_optimize_trees_for_given_msa(get_local_path(msa_path), ll_on_data_prefix = "iqtree_raxml", tree_file=trees_path,
                                           curr_run_directory=curr_run_directory, msa_type=msa_type, opt_brlen=True
                                           )

        unique_trees["final_ll"] = trees_ll_on_data
        msa_data = msa_data.drop(columns = ['final_ll','final_tree_topology'])
        msa_data = msa_data.merge(unique_trees, on  = ['tree_clusters_ind'] )

        msa_data["best_tree_ll_per_file"] = msa_data.groupby('file_name')['final_ll'].transform('max')

    best_msa_ll = max(msa_data["final_ll"])
    msa_data = msa_data.sort_values(["starting_tree_type", "starting_tree_ind", "spr_radius", "spr_cutoff"])
    best_msa_tree_topologies = list((msa_data[msa_data["final_ll"] == best_msa_ll].groupby('tree_clusters_ind')['final_tree_topology'].first()))
    msa_data["best_msa_ll"] = np.float(best_msa_ll)
    msa_data["rf_from_overall_msa_best_topology"] = msa_data["final_tree_topology"].apply(
        lambda x: min_rf_distance(curr_run_directory, x, best_msa_tree_topologies, name="MSA_enrichment_RF_calculations"))
    msa_data["final_ll"] = msa_data.groupby("tree_clusters_ind")["final_ll"].transform(max)
    msa_data["delta_ll_from_overall_msa_best_topology"] = np.where(
        (msa_data["rf_from_overall_msa_best_topology"]) > 0, best_msa_ll - msa_data["final_ll"], 0)
    return msa_data, msa_type


def unify_raw_data_csvs(raw_data_folder):
    csv_files_in_folder = [os.path.join(raw_data_folder, f) for f in
                           os.listdir(raw_data_folder) if f.endswith(CSV_SUFFIX)]
    dfs_in_folder = []
    for f in csv_files_in_folder:
        try:
            if LOCAL_RUN:
                data = pd.read_csv(f, sep=CSV_SEP) #
                print(data['msa_path'].unique())
            else:
                data = pd.read_csv(f, sep=CSV_SEP)
            data['file_name'] = os.path.basename(f)
            dfs_in_folder.append(data)
        except:
            pass
    logging.info(f"Combining CSV files: {csv_files_in_folder}")
    raw_data = pd.concat(dfs_in_folder, sort=False)

    #raw_data = raw_data.loc[raw_data.msa_path=='/groups/pupko/noaeker/data/New_MSAs/Selectome_msas/msas/ENSGT00940000154914.Euteleostomi.002.aa_masked.fas']
    return raw_data