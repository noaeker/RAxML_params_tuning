from help_functions import *
from datetime import datetime
from trees_manipulation import *
from config import *



def generate_raxml_ng_command_prefix(cpus=1):
    raxml_parallel_command = " --threads {N} --workers auto ".format(
        N=cpus)  # " --threads auto{{{N}}} --workers auto ".format(N=cpus)
    return raxml_parallel_command

class GENERAL_RAXML_ERROR(Exception):
    pass

def extract_raxml_statistics_from_msa(msa_path,msa_type, output_name, curr_run_directory):

    parsimony_tree_generation_prefix = os.path.join(curr_run_directory, output_name + "pars")
    constant_branch_length_parsimony_tree_path = parsimony_tree_generation_prefix + ".raxml.startTree"
    model = "GTR+G" if msa_type == "DNA" else "WAG+G"
    parsimony_tree_generation_command = (
        "{raxml_exe_path} {threads_config} --force msa --force perf_threads --start --msa {msa_path} --model {model} --tree pars{{{n_parsimony_trees}}} --seed {seed} --prefix {prefix}").format(
        raxml_exe_path=RAXML_NG_EXE,
        threads_config=generate_raxml_ng_command_prefix(),
        msa_path=msa_path, n_parsimony_trees=1, prefix=parsimony_tree_generation_prefix, seed=SEED, model =model)
    execute_commnand_and_write_to_log(parsimony_tree_generation_command)
    parsimony_model_evaluation_prefix = os.path.join(curr_run_directory, output_name + "pars_eval")
    parsimony_model_and_bl_evaluation_command = (
        "{raxml_exe_path} {threads_config} --force msa --force perf_threads --evaluate --msa {msa_path} --model {model}  --tree {parsimony_tree_path} --seed {seed} --prefix {prefix}").format(
        raxml_exe_path=RAXML_NG_EXE,
        threads_config=generate_raxml_ng_command_prefix(),
        msa_path=msa_path, parsimony_tree_path=constant_branch_length_parsimony_tree_path, seed=SEED,
        prefix=parsimony_model_evaluation_prefix, model =model)
    execute_commnand_and_write_to_log(parsimony_model_and_bl_evaluation_command)
    parsimony_log_path = parsimony_model_evaluation_prefix + ".raxml.log"
    parsimony_optimized_tree_path = parsimony_model_evaluation_prefix + ".raxml.bestTree"
    parsimony_tree_alpha = extract_param_from_raxmlNG_log(parsimony_log_path, "alpha")
    return {"parsimony_tree_path" : parsimony_optimized_tree_path,"parsimony_tree_alpha" : parsimony_tree_alpha }


def extract_mad_file_statistic(mad_log_path):
        pattern = "MAD=([\d.]+)"
        with open(mad_log_path) as mad_output:
            data = mad_output.read()
            match = re.search(pattern, data, re.IGNORECASE)
        if match:
            value = float(match.group(1))
        else:
            error_msg = "Param  not found in mad file in {}".format(mad_log_path)
            logging.error(error_msg)
            raise GENERAL_RAXML_ERROR(error_msg)
        return value




def extract_param_from_raxmlNG_log(raxml_log_path, param_name, raise_error=True):
    with open(raxml_log_path) as raxml_log_file:
        data = raxml_log_file.read()
        if (param_name == "alpha"):
            pattern = r'alpha: ([\d.]+)'
        elif (param_name == "search_ll"):
            pattern = r'Final LogLikelihood: (-[\d.]+)'
        elif (param_name == "rf_dist"):
            pattern = 'Average relative RF distance in this tree set: ([\d.]+)'
        elif (param_name == "time"):
            pattern = 'Elapsed time: ([\d.]+)'
        elif (param_name == "spr_cutoff"):
            pattern = 'spr subtree cutoff: ([\d.]+)'
        elif (param_name == "n_parsimony"):
            pattern = 'parsimony \((\d+)\)'
        elif (param_name == "n_random"):
            pattern = 'start tree\(s\): random \((\d+)\)'
        elif (param_name=="starting_tree_ll"):
            pattern = '\[\d{2}:\d{2}:\d{2} (-[\d.]+)\] Model parameter optimization'
        elif (param_name == "spr_radius"):
            pattern = 'SPR radius for FAST iterations: ([\d]+) '
            value_strings = re.findall(pattern, data)
            value_floats = [float(ll) for ll in value_strings]
            if len(value_floats) == 1:
                return value_floats[0]
            else:
                # logging.info("{} ll values were extracted from log file".format(len(value_floats)))
                return value_floats
        elif (param_name == "ll"):
            pattern = r'Tree #\d+, final logLikelihood: (-[\d.]+)'
            value_strings = re.findall(pattern, data)
            value_floats = [float(ll) for ll in value_strings]
            if len(value_floats) == 1:
                return value_floats[0]
            else:
                # logging.info("{} ll values were extracted from log file".format(len(value_floats)))
                return value_floats
        match = re.search(pattern, data, re.IGNORECASE)
        if match:
            value = float(match.group(1))
            return value
        else:
            error_msg = "Param {param_name} not found in file".format(param_name=param_name)
            if raise_error:
                raise GENERAL_RAXML_ERROR(error_msg)
            else:
                return None

def wait_for_file_existence(path, name):
    if not os.path.exists(path):
        # logging.info("{name} was succesfully created in: {path}".format(name=name, path=path))
        error_msg = "{name} was not generated in: {path}".format(name=name, path=path)
        logging.error(error_msg)
        start_time = time.time()
        while not os.path.exists(path):
            time.sleep(WAITING_TIME_UPDATE)
            logging.info("current time {}: file {} does not exist yet in path {}".format(datetime.now(), name, path))
            time.sleep(WAITING_TIME_UPDATE)
            if time.time() - start_time > 3600*24:
                logging.info("Waiting to much for param {}, breaking".format(name))
                break
        raise GENERAL_RAXML_ERROR(error_msg)






def filter_unique_topologies(curr_run_directory, trees_path, n):
    logging.debug("Removing duplicate SPR neighbours")
    rf_prefix = os.path.join(curr_run_directory, "SPR_neighbours")
    rf_command = (
        "{raxml_exe_path} --force msa --force perf_threads --rfdist --tree {rf_file_path} --prefix {prefix}").format(
        raxml_exe_path=RAXML_NG_EXE, rf_file_path=trees_path, prefix=rf_prefix)
    execute_commnand_and_write_to_log(rf_command)
    rf_distances_file_path = rf_prefix + ".raxml.rfDistances"
    unique_file_path = trees_path + "_unique"
    unique_topology_inds = set(list(range(n)))
    with open(rf_distances_file_path, 'r') as DIST, open(trees_path, 'r') as TREES, open(unique_file_path,
                                                                                         'w') as UNIQUE_TREES:
        distances = DIST.readlines()
        original_trees = TREES.readlines()
        for line in distances:
            lst = line.split("\t")
            curr_tree, comp_tree, dist = int(lst[0]), int(lst[1]), int(lst[2])
            if curr_tree in unique_topology_inds and comp_tree in unique_topology_inds and dist == 0:
                unique_topology_inds.remove(comp_tree)
        unique_trees = [original_trees[ind] for ind in unique_topology_inds]
        n_unique_top = len(unique_trees)
        logging.info(f'Found {n_unique_top} unique topologies')
        UNIQUE_TREES.writelines(unique_trees)
    rf_prefix = os.path.join(curr_run_directory, "SPR_neighbours_check")
    rf_command = (
        "{raxml_exe_path} --force msa --force perf_threads --rfdist --tree {rf_file_path} --prefix {prefix}").format(
        raxml_exe_path=RAXML_NG_EXE, rf_file_path=unique_file_path, prefix=rf_prefix)
    execute_commnand_and_write_to_log(rf_command)
    return unique_file_path

def generate_n_tree_topologies(n, original_file_path, curr_run_directory,
                               curr_msa_stats, seed, tree_type):
    prefix = os.path.join(curr_run_directory, f"{tree_type}_tree_generation")
    model = "GTR+G" if curr_msa_stats["msa_type"] == "DNA" else "WAG+G"
    random_tree_generation_command = (
        "{raxml_exe_path} {threads_config} --force msa --force perf_threads  --msa {msa_path} --model {model} --start --tree {tree_type}{{{n}}} --prefix {prefix} --seed {seed} ").format(
        n=n, raxml_exe_path=RAXML_NG_EXE, tree_type=tree_type,
        threads_config=generate_raxml_ng_command_prefix(cpus=curr_msa_stats["n_cpus_raxml"]),
        msa_path=original_file_path,  prefix=prefix, seed=seed, model = model)
    trees_path = prefix + ".raxml.startTree"
    raxml_log_file = prefix + ".raxml.log"
    execute_commnand_and_write_to_log(random_tree_generation_command,curr_msa_stats["print_commands_to_log"])
    elapsed_running_time = extract_param_from_raxmlNG_log(raxml_log_file, 'time')
    if tree_type=="pars" and n>1:
        logging.info("Removing duplicates parismony topologies")
        rf_prefix = os.path.join(curr_run_directory, "parsimony_rf_eval")
        rf_command = (
            "{raxml_exe_path} --force msa --force perf_threads --rfdist --tree {rf_file_path} --prefix {prefix}").format(
            raxml_exe_path=RAXML_NG_EXE, rf_file_path=trees_path, prefix=rf_prefix)
        execute_commnand_and_write_to_log(rf_command,curr_msa_stats["print_commands_to_log"])
        rf_distances_file_path = rf_prefix + ".raxml.rfDistances"
        trees_path = extract_parsimony_unique_topologies(curr_run_directory, trees_path,
                                                               rf_distances_file_path, n)
    return trees_path, elapsed_running_time

def extract_parsimony_unique_topologies(curr_run_directory, trees_path, dist_path, n):
    rf_prefix = os.path.join(curr_run_directory, "parsimony_rf")
    rf_command = (
        "{raxml_exe_path} --force msa --force perf_threads --rfdist --tree {rf_file_path} --prefix {prefix}").format(
        raxml_exe_path=RAXML_NG_EXE, rf_file_path=trees_path, prefix=rf_prefix)
    execute_commnand_and_write_to_log(rf_command)
    unique_file_path = trees_path + "_unique"
    unique_topology_inds = set(list(range(n)))
    with open(dist_path, 'r') as DIST, open(trees_path, 'r') as TREES, open(unique_file_path, 'w') as UNIQUE_TREES:
        distances = DIST.readlines()
        original_trees = TREES.readlines()
        for line in distances:
            lst = line.split("\t")
            curr_tree, comp_tree, dist = int(lst[0]), int(lst[1]), int(lst[2])
            if curr_tree in unique_topology_inds and comp_tree in unique_topology_inds and dist == 0:
                unique_topology_inds.remove(comp_tree)
        unique_trees = [original_trees[ind] for ind in unique_topology_inds]
        n_unique_top = len(unique_trees)
        logging.info(f'Found {n_unique_top} unique topologies')
        UNIQUE_TREES.writelines(unique_trees)
    rf_prefix = os.path.join(curr_run_directory, "parsimony_check_rf")
    rf_command = (
        "{raxml_exe_path} --force msa --force perf_threads --rfdist --tree {rf_file_path} --prefix {prefix}").format(
        raxml_exe_path=RAXML_NG_EXE, rf_file_path=unique_file_path, prefix=rf_prefix)
    execute_commnand_and_write_to_log(rf_command)
    return unique_file_path

def raxml_search(curr_msa_stats,curr_run_directory, msa_path, prefix,params_config, cpus, starting_tree_path):
    spr_radius = params_config.get("spr_radius")
    spr_cutoff = params_config.get("spr_cutoff")
    spr_radius_command = "--spr-radius {}".format(spr_radius) if spr_radius else ""
    spr_cutoff_command = "--spr-cutoff {}".format(spr_cutoff) if spr_cutoff else ""
    starting_trees_command = f"--tree {starting_tree_path} "
    search_prefix = os.path.join(curr_run_directory, prefix)
    model = "GTR+G" if curr_msa_stats["msa_type"] == "DNA" else "WAG+G"
    search_command = (
        "{raxml_exe_path}  {threads_config} --force msa --force perf_threads --msa {msa_path} --model {model} {starting_trees_command}  {spr_radius_command} {spr_cutoff_command} --seed {seed} --prefix {prefix}  ").format(
        raxml_exe_path=RAXML_NG_EXE,
        threads_config=generate_raxml_ng_command_prefix(cpus),
        msa_path=msa_path, starting_trees_command=starting_trees_command, seed=SEED,
        prefix=search_prefix, spr_radius_command = spr_radius_command, spr_cutoff_command = spr_cutoff_command, model = model)
    raxml_log_file = search_prefix + ".raxml.log"
    execute_commnand_and_write_to_log(search_command,curr_msa_stats["print_commands_to_log"])
    elapsed_running_time = extract_param_from_raxmlNG_log(raxml_log_file, 'time')
    best_ll = extract_param_from_raxmlNG_log(raxml_log_file, 'search_ll')
    starting_tree_ll  = extract_param_from_raxmlNG_log(raxml_log_file, 'starting_tree_ll')
    best_tree_topology_path = search_prefix + ".raxml.bestTree"
    actual_spr_radius = extract_param_from_raxmlNG_log(raxml_log_file, 'spr_radius')
    actual_spr_cutoff = extract_param_from_raxmlNG_log(raxml_log_file, 'spr_cutoff')
    res = {'spr_radius' : actual_spr_radius ,'spr_cutoff' : actual_spr_cutoff,'final_ll': best_ll, 'starting_tree_ll' : starting_tree_ll,
                'elapsed_running_time': elapsed_running_time,'final_tree_topology': get_tree_string(best_tree_topology_path)}
    return res

def raxml_optimize_trees_for_given_msa(curr_msa_stats,full_data_path, ll_on_data_prefix, tree_file, msa_stats,
                                       curr_run_directory, opt_brlen=True, weights=None, return_trees_file=False,
                                       n_cpus=1):

    curr_run_directory = os.path.join(curr_run_directory, ll_on_data_prefix)
    if os.path.exists(curr_run_directory):
        delete_dir_content(curr_run_directory)
    else:
        os.mkdir(curr_run_directory)
    logging.debug("RaxML: Evaluating likelihood on : " + full_data_path)
    alpha = msa_stats["alpha"]
    weights_path_command = "--site-weights {}".format(weights) if weights else ""
    logging.debug(
        "Optimizing branch lengths and using existing Gamma shape parameter: alpha={alpha}".format(alpha=alpha))
    prefix = os.path.join(curr_run_directory, ll_on_data_prefix)
    brlen_command = "--opt-branches off --opt-model off " if not opt_brlen else ""
    model = "GTR+G" if curr_msa_stats["msa_type"] == "DNA" else "WAG+G"
    compute_ll_run_command = (
        "{raxml_exe_path} {threads_config} --force msa --force perf_threads --evaluate --msa {msa_path} --model {model} {brlen_command} --tree {tree_file} {weights_path_command} --seed {seed} --prefix {prefix}").format(
        raxml_exe_path=RAXML_NG_EXE,
        threads_config=generate_raxml_ng_command_prefix(n_cpus),
        alpha=alpha, msa_path=full_data_path, tree_file=tree_file, seed=SEED,
        prefix=prefix, weights_path_command=weights_path_command, brlen_command=brlen_command, model = model)
    optimized_trees_path = prefix + ".raxml.mlTrees"
    best_tree_path = prefix + ".raxml.bestTree"
    raxml_log_file = prefix + ".raxml.log"
    execute_commnand_and_write_to_log(compute_ll_run_command,curr_msa_stats["print_commands_to_log"])

    trees_ll_on_data = extract_param_from_raxmlNG_log(raxml_log_file, "ll")
    elapsed_running_time = extract_param_from_raxmlNG_log(raxml_log_file, 'time')
    optimized_trees_final_path = optimized_trees_path if os.path.exists(optimized_trees_path) else best_tree_path
    tree_objects = generate_multiple_tree_object_from_newick(optimized_trees_final_path)
    if return_trees_file:
        return trees_ll_on_data, optimized_trees_final_path, elapsed_running_time
    return trees_ll_on_data, tree_objects, elapsed_running_time