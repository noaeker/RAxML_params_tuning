from side_code.help_functions import *
from side_code.config import *
import shutil
import pickle
from side_code.axml import *
from side_code.basic_trees_manipulation import *
from side_code.msa_runs import *


def generate_results_folder(curr_run_prefix):
    create_dir_if_not_exists(RESULTS_FOLDER)
    curr_run_prefix = os.path.join(RESULTS_FOLDER, curr_run_prefix)
    create_dir_if_not_exists(curr_run_prefix)
    return curr_run_prefix


def distribute_MSAs_over_jobs(all_jobs_results_folder, args, all_single_raxml_runs_dict):
    job_tracking_dict = {}
    run_keys = all_single_raxml_runs_dict.keys()
    runs_keys_splits = np.array_split(run_keys, args.n_jobs)
    for job_ind in range(len(args.n_jobs)):
        curr_job_runs_dict = {(k, all_single_raxml_runs_dict[k]) for k in runs_keys_splits[job_ind]}
        curr_job_folder = os.path.join(all_jobs_results_folder, "job_" + str(job_ind))
        create_or_clean_dir(curr_job_folder)
        curr_job_related_files_paths = get_job_related_files_paths(curr_job_folder, job_ind)
        curr_job_local_dump_path = curr_job_related_files_paths["job_local_raxml_runs_path"]
        curr_job_log_path = os.path.join(curr_job_folder, str(job_ind) + "_tmp_log")
        job_tracking_dict[job_ind] = {'job_log_path': curr_job_log_path, 'job_local_dump_path': curr_job_local_dump_path }
        pickle.dump(curr_job_runs_dict, open(curr_job_local_dump_path, "wb"))
        run_command = f' python {MAIN_CODE_PATH} --job_ind {job_ind} --curr_job_folder {curr_job_folder} --curr_job_local_dump_path {curr_job_local_dump_path}  {generate_argument_str(args)}'
        job_name = args.jobs_prefix + str(job_ind)
        if not LOCAL_RUN:
            submit_linux_job(job_name, curr_job_folder,curr_job_log_path, run_command, args.n_cpus_per_job, job_ind, queue=args.queue)
        else:
            submit_local_job(MAIN_CODE_PATH, ["--job_ind", str(job_ind), "--curr_job_folder", curr_job_folder, "--curr_job_local_dump_path", curr_job_local_dump_path
                                              ] + generate_argument_list(args))
    return job_tracking_dict


def update_global_raxml_runs_dict(job_tracking_dict, all_raxml_runs_global_dict):
    for job_ind in job_tracking_dict:
        for file in (job_tracking_dict[job_ind]["job_log_path"]):
            if (file.endswith('Err') or LOCAL_RUN):
                job_raxml_runs_dict = pickle.load(job_tracking_dict[job_ind]["job_local_dump_path"])
                all_raxml_runs_global_dict.update(job_raxml_runs_dict)
            del job_tracking_dict[job_ind]



def generate_global_raxml_runs_dict(args,all_jobs_results_folder):
    file_path_list = extract_alignment_files_from_dirs(args.general_msa_dir)
    if LOCAL_RUN:
        file_path_list = file_path_list[:100]
    logging.info("There are overall {nMSAs} available MSAs ".format(nMSAs=len(file_path_list)))
    file_path_list_full = remove_MSAs_with_not_enough_seq_and_locis(file_path_list, args.min_n_seq, args.min_n_loci)
    logging.debug("Alignment files are " + str(file_path_list))
    random.seed(SEED)
    file_path_list = random.sample(file_path_list_full, args.n_MSAs)
    logging.info(
        "There are {} MSAs with at least {} sequences and {} positions".format(len(file_path_list), args.min_n_seq,
                                                                               args.min_n_loci))
    # file_path_list = file_path_list[args.first_msa_ind:(args.first_msa_ind + args.n_MSAs)]
    logging.info(
        f"Sampling {args.n_MSAs} random MSAs")

    global_raxml_runs_dict = generate_all_single_raxml_runs(file_path_list, spr_radius_grid_str=args.spr_radius_grid,
                                                            spr_cutoff_grid_str=args.spr_cutoff_grid,
                                                            n_parsimony_tree_objects_per_msa=args.n_raxml_parsimony_trees,
                                                            n_random_tree_objects_per_msa=args.n_raxml_random_trees,
                                                            curr_run_directory=all_jobs_results_folder, seed=SEED)

    return global_raxml_runs_dict




def main():
    parser = main_parser()
    args = parser.parse_args()
    all_jobs_results_folder = generate_results_folder(args.run_prefix)
    all_jobs_general_log_file = os.path.join(all_jobs_results_folder, "log_file.log")
    logging.basicConfig(filename=all_jobs_general_log_file, level=LOGGING_LEVEL)
    logging.info("Args = {args}".format(args=args))
    logging.info('#Started running')

    general_data_dict_path = os.path.join(RESULTS_FOLDER, "data.dump")
    if os.path.exists(general_data_dict_path):
        global_raxml_runs_dict= pickle.load(general_data_dict_path)

    else:
        global_raxml_runs_dict = generate_global_raxml_runs_dict(args,all_jobs_results_folder)
        pickle.dump(global_raxml_runs_dict, open(general_data_dict_path, "wb"))
        logging.info(f"Saving all required raxml runs to {general_data_dict_path}")

    job_tracking_dict = distribute_MSAs_over_jobs(all_jobs_results_folder, args,global_raxml_runs_dict)
    while len(job_tracking_dict) > 0:
        t = time.localtime()
        current_time = time.strftime("%m/%d/%Y, %H:%M:%S", t)
        logging.debug(f"Current time {current_time}\nNumber of active jobs : {len(job_tracking_dict)}")
        time.sleep(WAITING_TIME_UPDATE)
        update_global_raxml_runs_dict(job_tracking_dict, global_raxml_runs_dict)
        pickle.dump(global_raxml_runs_dict, open(general_data_dict_path, "wb"))




if __name__ == "__main__":
    main()
