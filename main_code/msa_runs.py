from side_code.raxml import *
from side_code.help_functions import *
import numpy as np
import pandas as pd
from side_code.basic_trees_manipulation import *
from side_code.config import *



class single_raxml_run:
    def __init__(self, msa_path, starting_tree_object, starting_tree_type, params_config, status):
        self.msa_path = msa_path
        self.starting_tree_object = starting_tree_object
        self.starting_tree_type = starting_tree_type
        self.params_config = params_config
        self.status = status

    def set_run_results(self, results):
        self.results = results
        self.status = 1


def get_n_unique_tree_objects(msa_path, curr_run_directory, msa_stats, tree_type, n):
    topologies_path, elapsed_time_p = generate_n_unique_tree_topologies_as_starting_trees(
        n=n * 2,
        original_file_path=msa_path,
        curr_run_directory=curr_run_directory,
        curr_msa_stats=msa_stats, seed=SEED,
        tree_type=tree_type)[:n]
    tree_objects = generate_multiple_tree_object_from_newick(topologies_path)[:n]
    return tree_objects


def generate_tree_type_raxml_runs(msa_path,n_tree_objects_per_msa,msa_type, tree_type,curr_run_directory, grid_points, seed):
    runs = []
    n = n_tree_objects_per_msa * 2 if tree_type=="pars" else n_tree_objects_per_msa
    trees_path = generate_n_unique_tree_topologies_as_starting_trees(n,
                                                                               msa_type, curr_run_directory,
                                                                               seed, tree_type, msa_type)
    tree_objects = generate_multiple_tree_object_from_newick( trees_path)[
                             :n_tree_objects_per_msa]
    for tree_object in tree_objects:
        for params_config in grid_points:
            runs.append(single_raxml_run(msa_path=msa_path, starting_tree_object=tree_object,
                                             starting_tree_type=tree_type, params_config=params_config, status=0))
    return runs



def generate_all_single_raxml_runs(MSAs_list,spr_radius_grid_str, spr_cutoff_grid_str, n_parsimony_tree_objects_per_msa,
                                   n_random_tree_objects_per_msa, curr_run_directory, seed):

    all_msa_runs = []
    param_grid_str = {"spr_radius": spr_radius_grid_str, "spr_cutoff": spr_cutoff_grid_str}
    param_grid_obj = get_param_obj(param_grid_str)
    for msa_path in MSAs_list:
        msa_type = get_msa_type(msa_path)
        parsimony_raxml_runs = generate_tree_type_raxml_runs(msa_path,n_parsimony_tree_objects_per_msa,msa_type,"pars",curr_run_directory, param_grid_obj, seed)
        random_raxml_runs = generate_tree_type_raxml_runs(msa_path, n_random_tree_objects_per_msa, msa_type,
                                                             "rand", curr_run_directory, param_grid_obj, seed)
        all_msa_runs.append(parsimony_raxml_runs)
        all_msa_runs.append(random_raxml_runs)
    raxml_runs_dict = {id: raxml_run for id,raxml_run in enumerate(all_msa_runs)}
    return raxml_runs_dict
