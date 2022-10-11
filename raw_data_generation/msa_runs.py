from side_code.raxml import *
from side_code.basic_trees_manipulation import *
from side_code.config import *
from side_code.MSA_manipulation import get_msa_type
from sklearn.model_selection import ParameterGrid
import numpy as np
import uuid
import random



class single_raxml_run:
    '''
    Class that represents a single run of RAxML
    '''
    def __init__(self, msa_path, starting_tree_object, starting_tree_type, starting_tree_ind, params_config, type):
        self.msa_path = msa_path
        self.starting_tree_object = starting_tree_object
        self.starting_tree_type = starting_tree_type
        self.starting_tree_ind = starting_tree_ind
        self.params_config = params_config
        self.type = type

    def set_run_results(self, results):
        self.results = results
        self.status = 1

    def transform_to_dict(self):
        raxml_run_results = {"msa_path": self.msa_path,
                            "starting_tree_object": self.starting_tree_object.write(format=1),
                            "starting_tree_type":self.starting_tree_type,
                            "starting_tree_ind": self.starting_tree_ind,
                            "type": self.type,
                            }
        raxml_run_results.update(self.params_config)
        raxml_run_results.update(self.results)
        return raxml_run_results





def get_n_unique_tree_objects(msa_path, curr_run_directory, msa_stats, tree_type, n):
    topologies_path, elapsed_time_p = generate_n_unique_tree_topologies_as_starting_trees(
        n=n * 2,
        original_file_path=msa_path,
        curr_run_directory=curr_run_directory,
        curr_msa_stats=msa_stats, seed=SEED,
        tree_type=tree_type)[:n]
    tree_objects = generate_multiple_tree_object_from_newick(topologies_path)[:n]
    return tree_objects


def generate_tree_type_raxml_runs(msa_path, n_tree_objects_per_msa, msa_type, tree_type, curr_run_directory,
                                  grid_points, seed):
    '''

    :param msa_path:
    :param n_tree_objects_per_msa:
    :param msa_type:
    :param tree_type:
    :param curr_run_directory:
    :param grid_points:
    :param seed:
    :return: Default and grid points
    '''
    runs = []
    n = n_tree_objects_per_msa * 2 if tree_type == "pars" else n_tree_objects_per_msa
    trees_path = generate_n_unique_tree_topologies_as_starting_trees(n,
                                                                     msa_path, curr_run_directory,
                                                                     seed, tree_type, msa_type)
    tree_objects = generate_multiple_tree_object_from_newick(trees_path)[
                   :n_tree_objects_per_msa]
    for starting_tree_ind,tree_object in enumerate(tree_objects):
        runs.append(single_raxml_run(msa_path=msa_path, starting_tree_object=tree_object,
                                     starting_tree_type=tree_type, starting_tree_ind = starting_tree_ind, params_config={}, type="default")) #Add default_run
        for params_config in grid_points:
            runs.append(single_raxml_run(msa_path=msa_path, starting_tree_object=tree_object,
                                         starting_tree_type=tree_type,starting_tree_ind = starting_tree_ind, params_config=params_config, type="non-default"))
    return runs


def generate_all_raxml_runs_per_msa(msa_paths, spr_radius_grid_str, spr_cutoff_grid_str,
                                    n_parsimony_tree_objects_per_msa,
                                    n_random_tree_objects_per_msa, curr_run_directory, seed):

    runs = {}
    param_grid_str = {"spr_radius": spr_radius_grid_str, "spr_cutoff": spr_cutoff_grid_str}
    param_grid_obj = get_param_obj(param_grid_str)
    for msa_path in msa_paths:
        msa_runs = {}
        msa_type = get_msa_type(msa_path)
        msa_parsimony_raxml_runs = generate_tree_type_raxml_runs(msa_path, n_parsimony_tree_objects_per_msa, msa_type,
                                                             "pars", curr_run_directory, param_grid_obj, seed)
        for i, msa_run in enumerate(msa_parsimony_raxml_runs):
            msa_runs[f"{msa_path}_parsimony_{i}"]  = msa_run
        create_or_clean_dir(curr_run_directory)
        msa_random_raxml_runs = generate_tree_type_raxml_runs(msa_path, n_random_tree_objects_per_msa, msa_type,
                                                          "rand", curr_run_directory, param_grid_obj, seed)
        for i, msa_run in enumerate(msa_random_raxml_runs):
            msa_runs[f"{msa_path}_random_{i}"] = msa_run
        create_or_clean_dir(curr_run_directory)
        runs[msa_path] = msa_runs

    return runs


def generate_test_msa_raxml_run(test_msa_path,curr_run_directory, seed):
    msa_type = get_msa_type(test_msa_path)
    msa_parsimony_raxml_runs = generate_tree_type_raxml_runs(test_msa_path, 1, msa_type,
                                                             "pars", curr_run_directory, {}, seed)
    return msa_parsimony_raxml_runs[0]



def str_to_linspace(str):
    linespace_nums = [float(n) for n in str.split("_")]
    return np.linspace(linespace_nums[0], linespace_nums[1], int(linespace_nums[2]))


def get_param_obj(param_grid_dict_str):
    param_grid_obj = {}
    for param_name in param_grid_dict_str:
        if param_grid_dict_str[param_name] != "default":
            linspace = [float(x) for x in str.split(param_grid_dict_str[param_name],"_")]
            param_grid_obj[param_name] = linspace
    param_obj = (ParameterGrid(param_grid_obj))
    return param_obj
