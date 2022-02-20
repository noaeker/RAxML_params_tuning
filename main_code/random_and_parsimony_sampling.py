
from help_functions import *
import pandas as pd
import pickle


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
    parser = random_and_parsimony_job_parser()
    args = parser.parse_args()
    data = pd.read_csv(args.raw_data_path, sep=CSV_SEP)
    job_related_file_paths = get_sampling_job_related_files_paths(args.curr_job_folder, args.job_ind)
    job_grid_points_file, general_log_path, job_csv_path, curr_job_status_file = \
        job_related_file_paths[
            "job_grid_points_file"], \
        job_related_file_paths[
            "general_log_path"], \
        job_related_file_paths[
            "job_csv_path"], \
        job_related_file_paths[
            "job_status_file"]
    parameter_grid = pickle.load( open(job_grid_points_file, "rb" ) )
    print(parameter_grid)

    get_average_best_results_among_a_tree_set_per_msa(
            data, parameter_grid, n_sample_points=args.n_sample_points, sampling_csv_path=job_csv_path)





if __name__ == "__main__":
    main()