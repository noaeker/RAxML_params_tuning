import pandas as pd
from config import *
from help_functions import *
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats


def get_average_best_results_among_a_tree_set_per_msa(data, n_parsimony_grid, n_random_grid, n_sample_points,
                                                      sampling_csv_path,
                                                      run_name=None):
    if run_name:
        data = data[data["run_name"] == run_name]
    data.loc[data['run_name'] == "default", 'spr_radius'] = "default"
    data.loc[data['run_name'] == "default", 'spr_cutoff'] = "default"
    seed = SEED
    first_insert = True
    for n_parsimony in n_parsimony_grid:
        for n_random in n_random_grid:
            current_configuration_results = pd.DataFrame()
            for i in range(n_sample_points):
                seed = seed + 1
                sampled_data_parsimony = data[data["tree_type"] == "parsimony"].groupby(
                    by=["msa_name", "run_name", "spr_radius", "spr_cutoff", "best_msa_ll"], group_keys= False).apply(lambda df:
                    df.sample(n=n_parsimony, random_state=seed))

                sampled_data_random = data[data["tree_type"] == "random"].groupby(
                    by=["msa_name", "run_name", "spr_radius", "spr_cutoff", "best_msa_ll"], group_keys= False).apply(lambda df:
                    df.sample(n=n_random, random_state=seed))
                sampled_data = pd.concat([sampled_data_parsimony, sampled_data_random])
                run_metrics = sampled_data.groupby(
                    by=["msa_name", "run_name", "spr_radius", "spr_cutoff", "best_msa_ll"]).agg(
                    {"delta_ll_from_overall_msa_best_topology":['min','mean'], 'elapsed_running_time': ['sum']})
                run_metrics.columns = ["curr_sample_Err","curr_sample_overall_Err","curr_sample_total_time"]
                run_metrics.reset_index(inplace = True)
                run_metrics["n_parsimony"] = n_parsimony
                run_metrics["n_random"] = n_random
                run_metrics["i"] = i
                run_metrics["Err"] = -run_metrics["curr_sample_Err"] / run_metrics["best_msa_ll"]
                if current_configuration_results.empty:
                    current_configuration_results = run_metrics.copy()
                else:
                    current_configuration_results = pd.concat([current_configuration_results, run_metrics])
            aggregated_current_results = current_configuration_results.groupby(
                by=["msa_name", "run_name", "spr_radius", "spr_cutoff", "n_parsimony", "n_random"]).agg(
                {'curr_sample_overall_Err':['mean'],'curr_sample_Err':['mean','std'],'curr_sample_total_time': ['mean','std']})
            aggregated_current_results.columns = ['mean_Err_overall','mean_Err','std_Err','mean_time', 'std_time']
            aggregated_current_results.reset_index(inplace = True)

            aggregated_current_results.to_csv(sampling_csv_path, mode='a', header=first_insert, sep=CSV_SEP)
            first_insert = False



def rank_configurations_vs_default(data):
    default_confg = data[
        (data["run_name"] == "default") & (data["n_parsimony"] == 10) & (data["n_random"] == 10)].copy()
    default_confg = default_confg.rename(columns={'mean_time': 'default_time', "mean_Err": "default_error"})[
        ["msa_name", "default_time", "default_error"]]
    enriched_data = pd.merge(data, default_confg, on=["msa_name"])
    enriched_data.sort_values(by=["msa_name", "mean_Err", "mean_time"], inplace=True)
    enriched_data["running_time_vs_default"] = enriched_data["default_time"] / enriched_data["mean_time"]
    enriched_data.groupby("msa_name").head(30).to_csv("tmp_test.csv")


def default_mistake_statistics(data):
    data = data[data["run_name"] == "default"].copy()
    data = data.groupby(["msa_name", "best_msa_ll"]).agg(
        {"delta_ll_from_overall_msa_best_topology": 'min', "elapsed_running_time": 'sum'}).reset_index()
    data["Err"] = -data["delta_ll_from_overall_msa_best_topology"] / data["best_msa_ll"]
    data = data.rename(columns={"elapsed_running_time": "default_total_time"})
    return data
    # sns.histplot(x = "Err", data = data, color="red", bins =6)
    # sns.histplot(x="Err", data=data[data["Err"]>0], color="red")
    # plt.figure()
    # plt.show()


def plot_default_performance(data):
    pass


def plot_results(data, plots_dir):
    data["total_trees"] = data["n_parsimony"] + data["n_random"]
    data = data.rename(columns={"n_parsimony": "p", "n_random": "r"})
    sns.relplot(x="mean_time", y="mean_Err", hue="total_trees",
                data=data)
    # plt.show()

    plt.figure()
    sns.relplot(x="mean_time", y="mean_Err", hue="run_name", size="total_trees",
                data=data)
    # plt.show()
    plt.tight_layout()
    # plt.savefig(f'{plots_dir}/sampling_results.png')

    plt.figure()


def main():
    overall_data_path = f"{RESULTS_FOLDER}/full_raxml_data.tsv"
    # plots_dir = 'plots_dir_new'
    sampling_csv_path = f"{RESULTS_FOLDER}/sampled_raxml_data.tsv"
    # if not os.path.exists(plots_dir):
    #    os.mkdir(plots_dir)
    data = pd.read_csv(overall_data_path, sep=CSV_SEP)

    if not os.path.exists(sampling_csv_path):
        get_average_best_results_among_a_tree_set_per_msa(
            data, n_parsimony_grid=range(11),
            n_random_grid=range(11), n_sample_points=15, sampling_csv_path = sampling_csv_path)
    res = pd.read_csv(sampling_csv_path)
    # rank_configurations_vs_default(res)


if __name__ == "__main__":
    main()
