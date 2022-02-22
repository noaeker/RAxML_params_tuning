from side_code.help_functions import *
from side_code.config import *
import pandas as pd

def get_average_results_on_default_configurations_per_msa(default_data, n_sample_points,
                                                          sampling_csv_path, seed
                                                          ):
    default_results = pd.DataFrame()
    for i in range(n_sample_points):
        seed = seed + 1
        sampled_data_parsimony = default_data[default_data["tree_type"] == "parsimony"].groupby(
            by=["msa_name"]).sample(n=10, random_state=seed)
        sampled_data_random = default_data[default_data["tree_type"] == "random"].groupby(
            by=["msa_name"]).sample(n=10, random_state=seed)
        sampled_data = pd.concat([sampled_data_parsimony, sampled_data_random])
        run_metrics = sampled_data.groupby(
            by=["msa_name", "best_msa_ll"]).agg(
            {"delta_ll_from_overall_msa_best_topology": ['min', 'mean'], 'elapsed_running_time': ['sum']})
        run_metrics.columns = ["curr_sample_Err", "curr_sample_overall_Err", "curr_sample_total_time"]
        run_metrics.reset_index(inplace=True)
        if default_results.empty:
            default_results = run_metrics.copy()
        else:
            default_results = pd.concat([default_results, run_metrics])

    aggregated_default_results = default_results.groupby(
        by=["msa_name"]).agg(
        {'curr_sample_overall_Err': ['mean'], 'curr_sample_Err': ['mean', 'std'],
         'curr_sample_total_time': ['mean', 'std']})

    aggregated_default_results.columns = ['default_mean_Err_overall', 'default_mean_Err', 'default_std_Err', 'default_mean_time', 'default_std_time']
    aggregated_default_results.reset_index(inplace=True)

    aggregated_default_results.to_csv(sampling_csv_path, sep=CSV_SEP)
    return aggregated_default_results


def main():
    default_sampling_csv_path = os.path.join(RESULTS_FOLDER, f"default_sampling{CSV_SUFFIX}")
    raw_data = pd.read_csv(f"{RESULTS_FOLDER}/full_raxml_data.tsv", sep = CSV_SEP)
    default_data = raw_data[raw_data["run_name"] == "default"]
    get_average_results_on_default_configurations_per_msa(default_data, n_sample_points=5,
                                                          sampling_csv_path=default_sampling_csv_path, seed=SEED)

if __name__ == "__main__":
        main()
