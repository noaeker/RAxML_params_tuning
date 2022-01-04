from help_functions import *
from datetime import datetime
from trees_manipulation import *

class GENERAL_RAXML_ERROR(Exception):
    pass

def extract_param_from_IQTREE_log(raxml_log_path, param_name, raise_error=True):
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


def generate_iqtree_command_prefix(cpus=1):
    raxml_parallel_command = " --threads {N} --workers auto ".format(
        N=cpus)  # " --threads auto{{{N}}} --workers auto ".format(N=cpus)
    return raxml_parallel_command




def iqtree_search(curr_run_directory, msa_path, prefix,params_config, cpus):
    spr_radius = params_config.get("spr_radius")
    spr_cutoff = params_config.get("spr_cutoff")
    n_parsimony_trees = params_config.get("n_parsimony")
    n_random_trees = params_config.get("n_random")
    spr_radius_command = "--spr-radius {}".format(spr_radius) if spr_radius else ""
    spr_cutoff_command = "--spr-cutoff {}".format(spr_cutoff) if spr_cutoff else ""
    starting_trees_command = "--tree pars{{{n_parsimony_trees}}},rand{{{n_random_trees}}}".format(
            n_parsimony_trees=n_parsimony_trees,
            n_random_trees=n_random_trees) if n_parsimony_trees and n_random_trees else ""
    search_prefix = os.path.join(curr_run_directory, prefix)
    search_command = (
        "{raxml_exe_path}  {threads_config} --force msa --force perf_threads --msa {msa_path} --model WAG+G {starting_trees_command}  {spr_radius_command} {spr_cutoff_command} --seed {seed} --prefix {prefix}").format(
        raxml_exe_path=RAXML_NG_EXE,
        threads_config=generate_iqtree_command_prefix(cpus),
        msa_path=msa_path, starting_trees_command=starting_trees_command, seed=SEED,
        prefix=search_prefix, spr_radius_command = spr_radius_command, spr_cutoff_command = spr_cutoff_command)
    raxml_log_file = search_prefix + ".raxml.log"
    execute_commnand_and_write_to_log(search_command, curr_run_directory, job_folder_name="raxml_search_job",
                                      job_name="raxml_search", log_file_path=raxml_log_file, cpus=cpus,
                                      run_locally=LOCAL_RUN)
    elapsed_running_time = extract_param_from_IQTREE_log(raxml_log_file, 'time')
    best_ll = extract_param_from_IQTREE_log(raxml_log_file, 'search_ll')
    best_tree_topology_path = search_prefix + ".raxml.bestTree"
    actual_spr_radius = extract_param_from_IQTREE_log(raxml_log_file, 'spr_radius')
    actual_spr_cutoff = extract_param_from_IQTREE_log(raxml_log_file, 'spr_cutoff')
    actual_parsimony = extract_param_from_IQTREE_log(raxml_log_file, 'n_parsimony')
    actual_random = extract_param_from_IQTREE_log(raxml_log_file, 'n_random')
    res = {'spr_radius' : actual_spr_radius ,'spr_cutoff' : actual_spr_cutoff,'n_parsimony': actual_parsimony,'n_random':  actual_random,'best_ll': best_ll,
                'elapsed_running_time': elapsed_running_time,'best_tree_topology_path': best_tree_topology_path}
    return res