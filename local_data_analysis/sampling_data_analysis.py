import pandas as pd
from config import *
from help_functions import *
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

def rank_configurations_vs_default(sampling_data):
    sampling_data["is_default_run"] = (sampling_data["run_name"] == "default") & (sampling_data["n_parsimony"] == 10) & (sampling_data["n_random"] == 10)
    sampling_data["mean_Err_normalized"] = -sampling_data["mean_Err"]/sampling_data["best_msa_ll"]
    default_confg = sampling_data[sampling_data["is_default_run"]].copy()
    default_confg = default_confg.rename(columns={'mean_time': 'default_time', "mean_Err_normalized": "default_Err_normalized", "mean_Err" : "default_Err"})[
        ["msa_name", "default_time", "default_Err_normalized","default_Err"]]
    enriched_data = pd.merge(sampling_data, default_confg, on=["msa_name"])
    enriched_data = enriched_data[(enriched_data["mean_Err_normalized"]<=enriched_data["default_Err_normalized"])& (enriched_data["is_default_run"]== False)]
    enriched_data.sort_values(by=["msa_name", "mean_time"], inplace=True)
    enriched_data["running_time_vs_default"] = enriched_data["default_time"] / enriched_data["mean_time"]
    enriched_data["trees_confg"] = enriched_data["n_parsimony"].astype(str)+"_"+enriched_data["n_random"].astype(str)
    res = enriched_data.groupby(["msa_name"]).first()
    print(np.median(res["running_time_vs_default"]))
    #print(res)
    #sns.histplot(y="trees_confg", data=res, color="red", bins=50)
    plt.show()
    #sns.histplot(x="spr_radius", data=res, color="red", bins=50)
    #plt.show()
    #sns.histplot(x="spr_cutoff", data=res, color="red", bins=50)
    #plt.show()
    sns.relplot(x="running_time_vs_default", y = "n_loci", data=res, color="red")
    plt.show()
    sns.histplot(x="running_time_vs_default", data=res, color="red", bins=50)
    plt.show()
    res.to_csv("summarized_results.csv")
    return enriched_data


def summarize_results_per_msa(raw_data):
    data = raw_data[["msa_name","best_msa_ll"]]
    data = data.drop_duplicates()
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_data_path', action='store', type=str, default = f"{RESULTS_FOLDER}/full_raxml_data.tsv")
    parser.add_argument('--out_csv_path', action='store', type=str, default=f"{RESULTS_FOLDER}/sampled_raxml_data_new.tsv")
    args = parser.parse_args()
    # plots_dir = 'plots_dir_new'
    raw_data = pd.read_csv(args.raw_data_path, sep=CSV_SEP)
    per_msa_data = summarize_results_per_msa(raw_data)
    sampling_data = pd.read_csv(args.out_csv_path, sep = CSV_SEP)
    sampling_data = pd.merge(sampling_data, per_msa_data, on = ["msa_name"])
    ranking_data = rank_configurations_vs_default(sampling_data)



if __name__ == "__main__":
    main()