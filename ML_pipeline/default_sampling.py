
from side_code.config import *
import pandas as pd
import os

def get_average_results_on_default_configurations_per_msa(default_data, n_sample_points,
                                                          sampling_csv_path, seed
                                                          ):
    default_results = pd.DataFrame()
    for i in range(n_sample_points):
        seed = seed + 1
        sampled_data_parsimony = default_data[default_data["starting_tree_type"] == "pars"].groupby(
            by=["msa_path"]).sample(n=10, random_state=seed)
        sampled_data_random = default_data[default_data["starting_tree_type"] == "rand"].groupby(
            by=["msa_path"]).sample(n=10, random_state=seed)
        sampled_data = pd.concat([sampled_data_parsimony, sampled_data_random])
        run_metrics = sampled_data.groupby(
            by=["msa_path", "best_msa_ll"]).agg(
            {"delta_ll_from_overall_msa_best_topology": ['min'], "is_global_max": ['max'], 'relative_time': ['sum']})
        run_metrics.columns = ["curr_sample_Err", "curr_sample_is_global_max", "curr_sample_total_time"]
        run_metrics.reset_index(inplace=True)
        if default_results.empty:
            default_results = run_metrics.copy()
        else:
            default_results = pd.concat([default_results, run_metrics])

    # aggregated_default_results = default_results.groupby(
    #     by=["msa_name"]).agg(
    #     {'curr_sample_overall_Err': ['mean'], 'curr_sample_Err': ['mean', 'std'],
    #      'curr_sample_total_time': ['mean', 'std']})
    #
    # aggregated_default_results.columns = ['default_mean_Err_overall', 'default_mean_Err', 'default_std_Err', 'default_mean_time', 'default_std_time']
    # aggregated_default_results.reset_index(inplace=True)

    default_results.to_csv(sampling_csv_path, sep=CSV_SEP)
    return default_results


def main():
    default_sampling_csv_path = os.path.join(RESULTS_FOLDER, f"{READY_RAW_DATA}/c_30_70/default_sampling{CSV_SUFFIX}")
    print(default_sampling_csv_path)
    raw_data = pd.read_csv(f"{READY_RAW_DATA}/c_30_70/ML_edited_features.tsv", sep = CSV_SEP)
    default_data = raw_data[raw_data["type"] == "default"]
    get_average_results_on_default_configurations_per_msa(default_data, n_sample_points=1000,
                                                          sampling_csv_path=default_sampling_csv_path, seed=SEED)

if __name__ == "__main__":
        main()
