
import pandas as pd
import numpy as np
from side_code.config import *
import seaborn as sns
import matplotlib.pyplot as plt


def main():
    enriched_data =  pd.read_csv('/Users/noa/Workspace/raxml_deep_learning_results/current_raw_results/global_csv_enriched.tsv', sep = CSV_SEP
                                 )
    first_msa = list(enriched_data["msa_path"].unique())[4]
    enriched_data["global_max"] = enriched_data["delta_ll_from_overall_msa_best_topology"]==0
    enriched_data["normalized_time"] = enriched_data["elapsed_running_time"]/enriched_data["test_norm_const"]
    enriched_data["params"] = enriched_data["spr_radius"].map(str) + '-' + enriched_data["spr_cutoff"].map(str) + '-' + enriched_data["starting_tree_type"].map(str)
    agg_result = enriched_data.groupby(["msa_path","spr_radius","spr_cutoff","starting_tree_type","params","type"]).aggregate(pct_global = ('global_max',np.mean), avg_time = ('normalized_time',np.mean)).reset_index()
    first_msa_data = agg_result[agg_result["msa_path"]==first_msa]
    required_accuracy = 0.999
    possible_n_trees = list(range(1,21))
    best_n_trees_data = pd.DataFrame()
    for index,row in first_msa_data.iterrows():
        pct_success = [1-((1-row["pct_global"])**n_trees) for n_trees in possible_n_trees]
        time = [row["avg_time"]*n_trees for n_trees in possible_n_trees]
        curr_data = pd.DataFrame({'n_trees' : possible_n_trees, 'pct_success' : pct_success,'time': time})
        curr_data["spr_radius"] = row["spr_radius"]
        curr_data["spr_cutoff"] = row["spr_cutoff"]
        curr_data["params"] = row["params"]
        curr_data["starting_tree_type"] = row["starting_tree_type"]
        curr_data["type"] = row["type"]
        best_n_trees_data = pd.concat([best_n_trees_data, curr_data])
    best_n_trees_data_filtered = best_n_trees_data[best_n_trees_data["pct_success"]>required_accuracy].sort_values('time').head(30)
    sns.lineplot(x = "time", y = "pct_success", data = best_n_trees_data, hue = "params")
    plt.show()


    print(agg_result)
if __name__ == "__main__":
        main()
