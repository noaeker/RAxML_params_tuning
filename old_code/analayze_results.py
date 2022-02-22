import pandas as pd
from config import *
from help_functions import *
import seaborn as sns
import matplotlib.pyplot as plt


# from lifelines import CoxPHFitter


def get_best_results_among_a_tree_set_per_iteration(data, n_trees_grid, n_iter, tree_type=None, run_name=None):
    if tree_type:
        data = data[data["tree_type"] == tree_type]
    if run_name:
        data = data[data["run_name"] == run_name]
    data.loc[data['run_name'] == "default", 'spr_radius'] = "default"
    data.loc[data['run_name'] == "default", 'spr_cutoff'] = "default"
    all_results_df = pd.DataFrame()
    for i in range(n_iter):
        for n_trees in n_trees_grid:
            grouped_data = data.groupby(by=["msa_name", "tree_type", "run_name", "spr_radius", "spr_cutoff"]).sample(
                n=n_trees, random_state=i
            ).groupby(
                by=["msa_name", "tree_type", "run_name", "spr_radius", "spr_cutoff"]).agg(
                {"delta_ll_from_curr_starting_tree_best_topology": 'min', "elapsed_running_time": 'sum'}).reset_index()
            grouped_data["n_trees"] = n_trees
            grouped_data["iteration"] = i
            if all_results_df.empty:
                all_results_df = grouped_data.copy()
            else:
                all_results_df = pd.concat([all_results_df, grouped_data])

    all_results_df = all_results_df.reset_index()
    return all_results_df


def agg_tree_sampling_iterations(all_results_df):
    all_results_df_both_trees_agg = all_results_df.groupby(
        ["run_name", "iteration", "msa_name", "spr_radius", "spr_cutoff", "n_trees"]).agg(
        {"delta_ll_from_curr_starting_tree_best_topology": 'min', "elapsed_running_time": 'sum'}).reset_index()
    all_results_df_both_trees_agg["tree_type"] = "both"
    all_results_df_both_trees_agg["n_trees"] = all_results_df_both_trees_agg["n_trees"] * 2

    col_order = ["run_name", "iteration", "msa_name", "spr_radius", "spr_cutoff",
                 "delta_ll_from_curr_starting_tree_best_topology", "elapsed_running_time", "n_trees", "tree_type"]
    all_results_df_both_trees_agg = all_results_df_both_trees_agg[col_order]
    all_results_df = all_results_df[col_order]
    new_data = pd.concat([all_results_df, all_results_df_both_trees_agg], ignore_index=True)
    new_data.to_csv('test.csv')
    all_results_df_agg = new_data.groupby(
        ["n_trees", "run_name", "tree_type", "msa_name", "spr_radius", "spr_cutoff"]).agg(
        {"delta_ll_from_curr_starting_tree_best_topology": 'mean', "elapsed_running_time": 'mean'}).reset_index()
    return all_results_df_agg


def plot_spr_radius_and_cutoff_interactions(data, plots_dir):
    data["C"] = data["spr_cutoff"]
    data["R"] = data["spr_radius"]
    data["Err"] = data["delta_ll_from_curr_starting_tree_best_topology"]
    data = data[data["run_name"] != "default"]
    sns.relplot(
        data=data, x="spr_radius", y="Err",
        col="C",
        hue="tree_type", kind="line"
    )
    plt.savefig(f'{plots_dir}/radius_change_plot_grid.png')
    plt.figure()
    # plt.show()

    sns.relplot(
        data=data, x="spr_cutoff", y="Err",
        col="R",
        hue="tree_type", kind="line"
    )
    plt.savefig(f'{plots_dir}/cutoff_change_plot_grid.png')
    plt.figure()
    # plt.show()


def plot_effect_of_taking_more_trees(all_results_df_agg, plots_dir):
    all_results_df_agg["C"] = all_results_df_agg["spr_cutoff"]
    all_results_df_agg["R"] = all_results_df_agg["spr_radius"]
    all_results_df_agg["n"] = all_results_df_agg["n_trees"]
    all_results_df_agg["Err"] = all_results_df_agg["delta_ll_from_curr_starting_tree_best_topology"]

    # sns.relplot(
    #     data=all_results_df_agg[all_results_df_agg["run_name"] != "default"], x="n_trees", y="Err", col="C",
    #     row="R",
    #     hue="tree_type", kind="line"
    # )
    # plt.savefig(f'{plots_dir}/grid_line_plot.png')
    # plt.figure()
    # # plt.show()
    sns.lineplot(x="n_trees", y="Err", hue="tree_type",
                 data=all_results_df_agg[all_results_df_agg["run_name"] == "default"])
    plt.savefig(f'{plots_dir}/default_line_plot.png')
    plt.figure()
    # plt.show()
    # sns.catplot(data=all_results_df_agg[all_results_df_agg["run_name"] !="default"], hue="tree_type", x="n_trees",
    #            y="Err", kind="violin",col="C",row="R")
    # plt.savefig(f'{plots_dir}/boxplot_grid_plot.png')
    # plt.figure()
    # plt.show()
    # sns.boxplot(data = all_results_df_agg[all_results_df_agg["run_name"] == "default"], hue="tree_type", x = "n_trees", y="Err")
    # plt.savefig(f'{plots_dir}/boxplot_default_plot.png')
    # plt.figure()
    # plt.show()


def plot_spr_radius_and_cutoff_grid_vs_default_run(data, plots_dir):
    data["C"] = data["spr_cutoff"]
    data["R"] = data["spr_radius"]
    data["Err"] = data["LL_relative_to_default"]
    data["time"] = data["relative_elapsed_time_to_default"]
    data = data[data["run_name"] != "default"]
    for spr_cutoff in data["spr_cutoff"].unique():
        for spr_radius in data["spr_radius"].unique():
            curr_data = data[(data["C"] == spr_cutoff) & (data["R"] == spr_radius)]
            sns.relplot(
                data=curr_data, x="time", y="Err",
                hue="tree_type"
            )

            plt.axvline(1, min(curr_data["Err"]), max(curr_data["Err"]))
            plt.axhline(0, 0, max(curr_data["time"]))
            plt.savefig(f'{plots_dir}/radius_{spr_radius}_cutoff_{spr_cutoff}_vs_default.png')
            plt.tight_layout()
            # plt.show()
            plt.figure()



def plot_per_starting_tree_data(per_starting_tree_data, plots_dir):
    sns.relplot(
        data=per_starting_tree_data, y="default_d_ll_curr_tree", x="default_to_best_running_time",
        hue="tree_type"
    )
    plt.axvline(1, min(per_starting_tree_data["default_d_ll_curr_tree"]),
                max(per_starting_tree_data["default_d_ll_curr_tree"]))
    plt.tight_layout()
    plt.savefig(f'{plots_dir}/best_runs_vs_default.png')
    plt.figure()
    # plt.show()


def plot_per_msa_data(per_starting_tree_data, plots_dir):
    sns.relplot(
        data=per_starting_tree_data, y="default_best_overall_delta_ll", x="default_to_best_overall_running_time"
    )
    plt.axvline(1, min(per_starting_tree_data["default_best_overall_delta_ll"]),
                max(per_starting_tree_data["default_best_overall_delta_ll"]))
    plt.tight_layout()
    plt.savefig(f'{plots_dir}/best_runs_vs_default_overall.png')
    plt.figure()
    #plt.show()


def get_summary_per_starting_tree(data):
    best_running_time_starting_tree = \
        data[(data["delta_ll_from_curr_starting_tree_best_topology"] == 0)].copy().groupby(
            ["msa_name", "tree_type", "starting_tree_ind"]).agg({'elapsed_running_time': min}).reset_index()[
            ["msa_name", "tree_type", "starting_tree_ind", "elapsed_running_time"]].rename(
            columns={"elapsed_running_time": "best_running_time"})
    default_data_metrics = data[data["run_name"] == "default"][[
        "msa_name", "starting_tree_ind", "tree_type", "elapsed_running_time", "final_ll",
        "delta_ll_from_curr_starting_tree_best_topology", "delta_ll_from_overall_msa_best_topology", "spr_radius",
        "spr_cutoff"]].copy().rename(
        columns={"elapsed_running_time": "default_running_time", "final_ll": "default_ll",
                 "delta_ll_from_curr_starting_tree_best_topology": "default_d_ll_curr_tree",
                 "spr_radius": "default_spr_radius", "spr_cutoff": "default_spr_cutoff"})
    per_starting_tree_data = pd.merge(best_running_time_starting_tree, default_data_metrics,
                                      on=["msa_name", "tree_type", "starting_tree_ind"])
    per_starting_tree_data["default_to_best_running_time"] = per_starting_tree_data["default_running_time"] / \
                                                             per_starting_tree_data["best_running_time"]
    return per_starting_tree_data


def get_summary_per_msa(data):
    best_running_time_per_msa = \
        data[(data["delta_ll_from_overall_msa_best_topology"] == 0)].copy().groupby(
            ["msa_name"]).agg({'elapsed_running_time': min}).reset_index()[
            ["msa_name", "elapsed_running_time"]].rename(columns={"elapsed_running_time": "best_running_time_overall"})
    default_data_results = data[data["run_name"] == "default"]
    best_default_ll = data[data["run_name"] == "default"].groupby(
        "msa_name").agg({"delta_ll_from_overall_msa_best_topology": 'min'}).rename(
        columns={"delta_ll_from_overall_msa_best_topology": "best_overall_default_ll"})
    best_default_ll_and_elapsed_time = pd.merge(default_data_results, best_default_ll, on=["msa_name"])
    default_data_most_accurate_results = best_default_ll_and_elapsed_time[
        best_default_ll_and_elapsed_time["delta_ll_from_overall_msa_best_topology"] == best_default_ll_and_elapsed_time[
            "best_overall_default_ll"]].rename(columns={"delta_ll_from_overall_msa_best_topology": "default_best_overall_delta_ll"})
    default_data_accurate_and_fastest = default_data_most_accurate_results.groupby(
        ["msa_name", "default_best_overall_delta_ll"]).agg({"elapsed_running_time": "min"}).reset_index().rename(columns={"elapsed_running_time": "best_default_elapsed_time"})
    per_msa_data = pd.merge(best_running_time_per_msa, default_data_accurate_and_fastest,
                                      on=["msa_name"])
    per_msa_data["default_to_best_overall_running_time"] =  per_msa_data["best_default_elapsed_time"]/per_msa_data["best_running_time_overall"]
    return per_msa_data


# def enrich_data(data,per_tree_search_data):
#     enriched_data = pd.merge(data,per_tree_search_data, how="inner", on=["msa_name", "starting_tree_ind", "tree_type"])
#     enriched_data["relative_elapsed_time_to_default"] = enriched_data["elapsed_running_time"] / enriched_data[
#         "default_running_time"]
#     enriched_data["overall_LL_relative_to_default"] = enriched_data["delta_ll_from_overall_msa_best_topology"] - enriched_data[
#         "default_d_ll_overall"]
#     enriched_data["local_LL_relative_to_default"] = enriched_data["delta_ll_from_curr_starting_tree_best_topology"] - \
#                                                       enriched_data[
#                                                           "default_d_ll_curr_tree"]
#     return enriched_data


def get_tree_sampling_statistics(data, tmp_csv_path, plots_dir):
    n_iter = 5
    n_trees_grid = list(range(1, 11))
    if os.path.exists(tmp_csv_path):
        all_results_df = pd.read_csv(tmp_csv_path, index_col=None)
    else:
        all_results_df = get_best_results_among_a_tree_set_per_iteration(data, n_trees_grid, n_iter=n_iter)
        all_results_df.to_csv(tmp_csv_path, index=False)
    all_results_df_agg = agg_tree_sampling_iterations(all_results_df)
    plot_effect_of_taking_more_trees(all_results_df_agg, plots_dir)


def main():
    overall_data_path = "/Users/noa/Workspace/raxml_deep_learning_results/single_gene_MSAs_new/full_raxml_data.tsv"
    plots_dir = 'plots_dir_new'
    if not os.path.exists(plots_dir):
        os.mkdir(plots_dir)
    data = pd.read_csv(overall_data_path, sep=CSV_SEP)
    plot_spr_radius_and_cutoff_interactions(data, plots_dir)
    per_starting_tree_statistics = get_summary_per_starting_tree(data)
    plot_per_starting_tree_data(per_starting_tree_statistics, plots_dir)
    per_msa_data = get_summary_per_msa(data)
    print(per_msa_data.shape)
    plot_per_msa_data(per_msa_data, plots_dir)
    tmp_csv_path = "tmp_results_new.csv"
    get_tree_sampling_statistics(data, tmp_csv_path, plots_dir)


if __name__ == "__main__":
    main()
