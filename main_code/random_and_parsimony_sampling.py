import pandas as pd
from config import *
from help_functions import *
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats


def get_average_best_results_among_a_tree_set_per_msa(data, n_parsimony_grid, n_random_grid, n_sample_points,
                                                      sampling_csv_path
                                                      ):
    seed = SEED
    first_insert = True
    for i in range(n_sample_points):
        seed = seed + 1
        n_parsimony = np.random.choice(n_parsimony_grid)
        parsimony_cutoff = np.random.choice(n_parsimony_grid)
        parsimony_radius = np.random.choice(n_parsimony_grid)
        n_random = np.random.choice(n_random_grid)
        random_cutoff = np.random.choice(n_random_grid)
        parsim_radius = np.random.choice(n_random_grid)
        sampled_data_parsimony = data[data["tree_type"] == "parsimony"].groupby(
            by=["msa_name", "run_name", "best_msa_ll"]).sample(n=n_parsimony, random_state=seed)

        sampled_data_random = data[data["tree_type"] == "random"].groupby(
            by=["msa_name", "run_name", "best_msa_ll"]).sample(n=n_random, random_state=seed)
        sampled_data = pd.concat([sampled_data_parsimony, sampled_data_random])
        run_metrics = sampled_data.groupby(
            by=["msa_name", "run_name", "spr_radius", "spr_cutoff", "best_msa_ll"]).agg(
            {"delta_ll_from_overall_msa_best_topology":['min','mean'], 'elapsed_running_time': ['sum']})
        run_metrics.columns = ["curr_sample_Err","curr_sample_overall_Err","curr_sample_total_time"]
        run_metrics.reset_index(inplace = True)
        run_metrics["n_parsimony"] = n_parsimony
        run_metrics["n_random"] = n_random
        run_metrics["i"] = i
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





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_data_path', action='store', type=str, default = f"{RESULTS_FOLDER}/full_raxml_data.tsv")
    parser.add_argument('--out_csv_path', action='store', type=str, default=f"{RESULTS_FOLDER}/sampled_raxml_data_test.tsv")
    parser.add_argument('--n_parsimony', action='store', type=int,
                        default=10)
    parser.add_argument('--n_random', action='store', type=int,
                        default=10)
    parser.add_argument('--n_sample_points', action='store', type=int,
                        default=15)
    parser.add_argument('--seed', action='store', type=int,
                        default=15)
    args = parser.parse_args()
    # if not os.path.exists(plots_dir):
    #    os.mkdir(plots_dir)
    data = pd.read_csv(args.raw_data_path, sep=CSV_SEP)
    if not os.path.exists(args.out_csv_path):
        get_average_best_results_among_a_tree_set_per_msa(
            data, n_parsimony_grid=range(args.n_parsimony+1),
            n_random_grid=range(args.n_random+1), n_sample_points=args.n_sample_points, sampling_csv_path = args.out_csv_path)



if __name__ == "__main__":
    main()
