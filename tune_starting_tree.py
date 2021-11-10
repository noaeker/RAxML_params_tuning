from raxml import *
import numpy as np
from sklearn.model_selection import ParameterGrid
from generate_training_data import *
from generate_SPR import *


def SPR_on_MSA(msa_stats, msa_path, curr_run_directory, starting_tree_path, n_cpus):
    logging.info(f" Running RAxML on msa in: {msa_path} ")
    spr_search_run_directory = os.path.join(curr_run_directory, "SPR_runs")
    create_dir_if_not_exists(spr_search_run_directory)
    SPR_search_results = SPR_search(msa_path, "SPR_run", msa_stats, starting_tree_path,
               curr_run_directory
              , n_cpus = n_cpus)
    return  SPR_search_results



def main():
    parser = job_parser()
    args = parser.parse_args()
    job_related_file_paths = get_job_related_files_paths(args.curr_job_folder, args.job_ind)
    job_msa_paths_file, general_log_path, job_csv_path, job_best_csv_path, curr_job_status_file = \
    job_related_file_paths[
        "job_msa_paths_file"], \
    job_related_file_paths[
        "general_log_path"], \
    job_related_file_paths[
        "job_csv_path"], \
    job_related_file_paths["job_only_best_csv_path"], \
    job_related_file_paths[
        "job_status_file"]
    with open(job_msa_paths_file, "r") as paths_file:
        curr_job_file_path_list = paths_file.read().splitlines()
    logging.basicConfig(filename=general_log_path, level=LOGGING_LEVEL)
    logging.info('#Started running on job' + str(args.job_ind))
    logging.info("Job arguments : {}".format(args))

    job_results = pd.DataFrame(
    )
    job_results.to_csv(job_csv_path, index=False)


    job_best_results = pd.DataFrame(
    )
    job_best_results.to_csv(job_best_csv_path, index=False)
    for file_ind, original_alignment_path in enumerate(curr_job_file_path_list):
        msa_name = original_alignment_path.replace(MSAs_FOLDER, "").replace("ref_msa.aa.phy", "").replace(os.path.sep,
                                                                                                          "_")
        logging.info(
            f'#running on file name {msa_name} and ind (relativ to job) {file_ind}  original path= {original_alignment_path}')
        curr_msa_folder = os.path.join(args.curr_job_folder, msa_name)
        create_or_clean_dir(curr_msa_folder)
        msa_stats = handle_msa(curr_msa_folder, original_alignment_path, args.n_seq, args.n_loci)
        msa_stats.update(vars(args))
        extract_raxml_statistics_from_msa(original_alignment_path, f"msa_{file_ind}", msa_stats,  curr_msa_folder)
        logging.info(f"Basic MSA stats {msa_stats}\n")
        for i in range (5):
            print(5)
        job_results = job_results.append(curr_msa_data_analysis, ignore_index=True)
        best_raxml_result = (curr_msa_data_analysis[ curr_msa_data_analysis["rf_from_best_topology"] == 0]).sort_values(
            'elapsed_running_time', ascending=True).head(1)
        job_best_results = job_best_results.append(best_raxml_result)
        job_results.to_csv(job_csv_path)
        job_best_results.to_csv(job_best_csv_path)

    with open(curr_job_status_file, 'w') as job_status_f:
        job_status_f.write("Done")
    logging.info("Current job is done")



if __name__ == "__main__":
    main()


