import pandas as pd
import numpy as np
from side_code.config import *
from math import ceil
import seaborn as sns
import matplotlib.pyplot as plt


def main():
    enriched_raw_data = pd.read_csv(
        '/Users/noa/Workspace/raxml_deep_learning_results/current_raw_results/global_csv_enriched_new.tsv', sep=CSV_SEP
        )
    enriched_raw_data["global_max"] = enriched_raw_data["delta_ll_from_overall_msa_best_topology"] == 0
    enriched_raw_data["normalized_time"] = enriched_raw_data["elapsed_running_time"] / enriched_raw_data[
        "test_norm_const"]

    enriched_raw_data_default = enriched_raw_data[enriched_raw_data["type"]=="default"]
    enriched_raw_data_non_default = enriched_raw_data[enriched_raw_data["type"]!="default"]

    required_accuracy = 0.99

    agg_data_non_default = enriched_raw_data_non_default.groupby(
        ["msa_path", "spr_radius", "spr_cutoff", "starting_tree_type", "type"]).aggregate(
        pct_global=('global_max', np.mean),median_err = ('delta_ll_from_overall_msa_best_topology'), avg_time=('normalized_time', np.mean)).reset_index()


    #agg_data_non_default["pct_global"] = agg_data_non_default["pct_global"].apply(lambda x: x if x > 0 else 0.001)
    agg_data_non_default["required_n_trees"] = np.ceil(
        np.log(1 - required_accuracy) / np.log((1 - agg_data_non_default["pct_global"]))).replace({0: 1})
    agg_data_non_default["actual_pct_success"] = 1 - ((1 - agg_data_non_default["pct_global"]) ** agg_data_non_default["required_n_trees"])
    agg_data_non_default["required_time"] = np.abs(agg_data_non_default["avg_time"] * agg_data_non_default["required_n_trees"])


    agg_data_default = enriched_raw_data_default.groupby(
        ["msa_path","starting_tree_type"]).aggregate(
        default_pct_global=('global_max', np.mean), default_avg_time=('normalized_time', np.mean)).reset_index()
    agg_data_default = (agg_data_default.pivot_table(index=['msa_path'],
                          columns=['starting_tree_type'],
                          values=['default_pct_global','default_avg_time'],
                          aggfunc='first'))
    agg_data_default.columns = ['_'.join(col) for col in agg_data_default.columns]
    agg_data_default = agg_data_default.reset_index()
    agg_data_default["default_final_pct_success"] = 1-((1-agg_data_default["default_pct_global_rand"]))**10*((1-agg_data_default["default_pct_global_pars"]))**10
    agg_data_default["default_final_time"] = 10*agg_data_default["default_avg_time_pars"]+10*agg_data_default["default_avg_time_rand"]
    agg_data_default = agg_data_default[["msa_path","default_final_pct_success","default_final_time"]]
    final_data_agg = agg_data_non_default.merge(agg_data_default, on = "msa_path")
    final_data_agg["min_time_per_msa"] = final_data_agg.groupby("msa_path")["required_time"].transform("min")
    final_data_agg["is_optimal"] = final_data_agg["min_time_per_msa"] == final_data_agg["required_time"]
    final_data_agg["running_time_ratio"] = final_data_agg["default_final_time"]/final_data_agg["min_time_per_msa"]
    only_optimal_final_data = final_data_agg[final_data_agg["is_optimal"]]
    pass

    final_data_agg.to_csv(
        '/Users/noa/Workspace/raxml_deep_learning_results/current_raw_results/global_csv_enriched_agg.tsv', sep=CSV_SEP)




if __name__ == "__main__":
    main()
