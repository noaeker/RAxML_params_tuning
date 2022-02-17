
from help_functions import *
import pandas as pd


def get_average_best_results_among_a_tree_set_per_msa(data, parameter_grid, n_sample_points,
                                                      sampling_csv_path
                                                      ):
    seed = SEED
    first_insert = True
    for config in parameter_grid:
        current_config_results = pd.DataFrame()
        for i in range(n_sample_points):
            seed = seed + 1
            sampled_data_parsimony = data[
                (data["tree_type"] == "parsimony") & (data["spr_radius"] == config["parsimony_spr_radius"]) & (
                        data["spr_cutoff"] == config["parsimony_spr_cutoff"])].groupby(
                by=["msa_name", "best_msa_ll"]).sample(n=config["n_parsimony"], random_state=seed)

            sampled_data_random = data[
                (data["tree_type"] == "random") & (data["spr_radius"] == config["random_spr_radius"]) & (
                        data["spr_cutoff"] == config["random_spr_cutoff"])].groupby(
                by=["msa_name", "best_msa_ll"]).sample(n=config["n_random"], random_state=seed)
            sampled_data = pd.concat([sampled_data_parsimony, sampled_data_random])
            run_metrics = sampled_data.groupby(
                by=["msa_name", "best_msa_ll"]).agg(
                {"delta_ll_from_overall_msa_best_topology": ['min', 'mean'], 'elapsed_running_time': ['sum']})
            run_metrics.columns = ["curr_sample_Err", "curr_sample_overall_Err", "curr_sample_total_time"]
            run_metrics.reset_index(inplace=True)
            run_metrics["n_parsimony"] = config["n_parsimony"]
            run_metrics["n_random"] = config["n_random"]
            run_metrics["parsimony_spr_radius"] = config["parsimony_spr_radius"]
            run_metrics["parsimony_spr_cutoff"] = config["parsimony_spr_cutoff"]
            run_metrics["random_spr_radius"] = config["random_spr_radius"]
            run_metrics["random_spr_cutoff"] = config["random_spr_cutoff"]
            run_metrics["i"] = i
            if current_config_results.empty:
                current_config_results = run_metrics.copy()
            else:
                current_config_results = pd.concat([current_config_results, run_metrics])
        aggregated_current_results = current_config_results.groupby(
            by=["msa_name", "parsimony_spr_radius", "parsimony_spr_cutoff", "random_spr_radius", "random_spr_cutoff",
                "n_parsimony", "n_random"]).agg(
            {'curr_sample_overall_Err': ['mean'], 'curr_sample_Err': ['mean', 'std'],
             'curr_sample_total_time': ['mean', 'std']})
        aggregated_current_results.columns = ['mean_Err_overall', 'mean_Err', 'std_Err', 'mean_time', 'std_time']
        aggregated_current_results.reset_index(inplace=True)

        aggregated_current_results.to_csv(sampling_csv_path, mode='a', header=first_insert, sep=CSV_SEP)
        first_insert = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_data_path', action='store', type=str, default=f"{RESULTS_FOLDER}/full_raxml_data.tsv")
    parser.add_argument('--out_csv_path', action='store', type=str,
                        default=f"{RESULTS_FOLDER}/sampled_raxml_data_test2.tsv")
    parser.add_argument('--n_parsimony', action='store', type=int,
                        default=2)
    parser.add_argument('--n_random', action='store', type=int,
                        default=2)
    parser.add_argument('--n_sample_points', action='store', type=int,
                        default=1)
    parser.add_argument('--seed', action='store', type=int,
                        default=SEED)
    args = parser.parse_args()
    # if not os.path.exists(plots_dir):
    #    os.mkdir(plots_dir)
    data = pd.read_csv(args.raw_data_path, sep=CSV_SEP)
    grid_data = data[data["run_name"] != "default"]
    random_spr_cutoff_options = np.unique(grid_data["spr_cutoff"])
    random_spr_radius_options = np.unique(grid_data["spr_radius"])
    parsimony_spr_cutoff_options = random_spr_cutoff_options.copy()
    parsimony_spr_radius_options = random_spr_radius_options.copy()
    n_random_grid = range(args.n_random + 1)
    n_parsimony_grid = range(args.n_parsimony + 1)
    parameter_grid = ParameterGrid(
        {"random_spr_cutoff": random_spr_cutoff_options, "random_spr_radius": random_spr_radius_options,
         "parsimony_spr_cutoff": parsimony_spr_cutoff_options, "parsimony_spr_radius": parsimony_spr_radius_options,
         "n_random": n_random_grid, "n_parsimony": n_parsimony_grid})

    if not os.path.exists(args.out_csv_path):
        get_average_best_results_among_a_tree_set_per_msa(
            data, parameter_grid, n_sample_points=args.n_sample_points, sampling_csv_path=args.out_csv_path)


if __name__ == "__main__":
    main()
