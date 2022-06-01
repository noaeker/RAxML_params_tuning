
import pandas as pd
import numpy as np
from side_code.config import *
import seaborn as sns
import matplotlib.pyplot as plt


def main():
    enriched_raw_data =  pd.read_csv('/Users/noa/Workspace/raxml_deep_learning_results/current_raw_results/global_csv_enriched_new.tsv', sep = CSV_SEP
                                 )
    enriched_raw_data["global_max"] = enriched_raw_data["delta_ll_from_overall_msa_best_topology"]==0
    enriched_raw_data["normalized_time"] = enriched_raw_data["elapsed_running_time"]/enriched_raw_data["test_norm_const"]
    enriched_raw_data["params"] = enriched_raw_data["spr_radius"].map(str) + '-' + enriched_raw_data["spr_cutoff"].map(str) + '-' + enriched_raw_data["starting_tree_type"].map(str)
    agg_result = enriched_raw_data.groupby(["msa_path","spr_radius","spr_cutoff","starting_tree_type","params","type"]).aggregate(pct_global = ('global_max',np.mean), avg_time = ('normalized_time',np.mean)).reset_index()
    possible_n_trees = list(range(1,21))
    per_trees_data = pd.DataFrame()
    for index,row in agg_result.iterrows():
        pct_success = [1-((1-row["pct_global"])**n_trees) for n_trees in possible_n_trees]
        time = [row["avg_time"]*n_trees for n_trees in possible_n_trees]
        curr_data = pd.DataFrame({'n_trees' : possible_n_trees, 'pct_success' : pct_success,'time': time})#'spr_radius':row["spr_radius"], 'spr_cutoff': row["spr_cutoff"], 'params': row["params"]})
        curr_data = curr_data.assign(**dict(row))
        per_trees_data = pd.concat([per_trees_data, curr_data])
    required_accuracy = 0.999
    per_trees_data = per_trees_data[per_trees_data["pct_success"]>required_accuracy]
    per_trees_data['best_time'] = per_trees_data.groupby(['msa_path'])['time'].transform(min)
    best_confg_per_msa = per_trees_data[per_trees_data['time']==per_trees_data['best_time']]
    agg_result.to_csv(
        '/Users/noa/Workspace/raxml_deep_learning_results/current_raw_results/global_csv_enriched_agg_new.tsv', sep=CSV_SEP
        )
    pass


    print(agg_result)


if __name__ == "__main__":
        main()
