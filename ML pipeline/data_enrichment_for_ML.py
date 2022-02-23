import sys
sys.append('../')
from side_code.help_functions import *
import pandas as pd


def summarize_results_per_msa(raw_data):
    data = raw_data[["msa_name", "best_msa_ll", "msa_type"]]
    data["msa_type_numeric"] = data["msa_type"] == "AA"
    data = data.drop_duplicates()
    return data


def enrich_sampling_data_with_defaults_and_msa_statistics(sampling_data, default_data, raw_data):
    per_msa_data = summarize_results_per_msa(raw_data)
    sampling_data_with_msa_statistics = pd.merge(sampling_data, per_msa_data, on=["msa_name"])
    sampling_data_vs_default = pd.merge(sampling_data_with_msa_statistics, default_data, on=["msa_name"])
    sampling_data_vs_default["mean_Err_normalized"] = sampling_data_vs_default["mean_Err"] / sampling_data_vs_default["best_msa_ll"]
    sampling_data_vs_default["default_mean_Err_normalized"] = sampling_data_vs_default["default_mean_Err"] / sampling_data_vs_default[
        "best_msa_ll"]

    sampling_data_vs_default["normalized_error_vs_default"] = sampling_data_vs_default["mean_Err_normalized"] - \
                                                              sampling_data_vs_default["default_mean_Err_normalized"]
    sampling_data_vs_default["running_time_vs_default"] = sampling_data_vs_default["default_mean_time"] / \
                                                          sampling_data_vs_default[
                                                              "mean_time"]
    sampling_data_vs_default = sampling_data_vs_default[sampling_data_vs_default["spr_radius"] != "default"]
    best_running_time = sampling_data_vs_default[sampling_data_vs_default["normalized_error_vs_default"] <= 0].groupby(
        ["msa_name"]).agg({'running_time_vs_default': 'max'}).rename(
        columns={"running_time_vs_default": "best_running_time_vs_default"})
    sampling_data_vs_default = pd.merge(sampling_data_vs_default, best_running_time, on=["msa_name"])

    sampling_data_vs_default["is_optimal_run"] = (sampling_data_vs_default["running_time_vs_default"] ==
                                                  sampling_data_vs_default["best_running_time_vs_default"]) & (
                                                         sampling_data_vs_default["normalized_error_vs_default"] <= 0)
    return sampling_data_vs_default


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_data_path', action='store', type=str, default=f"{RESULTS_FOLDER}/full_raxml_data{CSV_SUFFIX}")
    parser.add_argument('--sampling_data', action='store', type=str,
                        default=f"{RESULTS_FOLDER}/sampled_raxml_data_large{CSV_SUFFIX}")
    parser.add_argument('--default_sampling_data', action='store', type=str,
                        default=f"{RESULTS_FOLDER}/default_sampling{CSV_SUFFIX}")
    parser.add_argument('--features_path', action='store', type=str,
                        default=f"{RESULTS_FOLDER}/features{CSV_SUFFIX}")
    parser.add_argument('--ML_dataset_output_path', action='store', type=str,
                        default=f"{RESULTS_FOLDER}/final_ML_dataset{CSV_SUFFIX}")
    args = parser.parse_args()
    raw_data = pd.read_csv(args.raw_data_path, sep=CSV_SEP)
    sampling_data =  pd.read_csv(args.sampling_data, sep=CSV_SEP)
    default_data = pd.read_csv(args.default_sampling_data, sep=CSV_SEP)
    features_data = pd.read_csv(args.features_path, sep=CSV_SEP)
    enriched_sampling_data_label = enrich_sampling_data_with_defaults_and_msa_statistics(sampling_data, default_data, raw_data)
    ML_dataset = pd.merge(enriched_sampling_data_label, features_data, on=["msa_name"])
    ML_dataset.to_csv(args.ML_dataset_output_path, sep = CSV_SEP)



if __name__ == "__main__":
        main()
