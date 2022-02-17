
from help_functions import *
from ete3 import *
import logging
import re
import numpy as np
from raxml import *

class Edge:
    def __init__(self, node_a, node_b):
        self.node_a = node_a
        self.node_b = node_b

    def __str__(self):
        return ("[a={a} b={b}]".format(a=self.node_a, b=self.node_b))

    def __eq__(self, other):
        """Overrides the default implementation"""
        if ((self.node_a == other.node_a) and (self.node_b == other.node_b)) or (
                (self.node_b == other.node_a) and (self.node_a == other.node_b)):
            return True
        else:
            return False


def print_subtree(tree, log_file, text):
    if log_file:
        log_file.write(text + " visualization: " + "\n" + tree.get_ascii(attributes=['name'],
                                                                         show_internal=True) + "\n")
        log_file.write(text + " newick " + str(tree.write(format=1)) + "\n")
    else:
        logging.info(text + " visualization: " + "\n" + tree.get_ascii(attributes=['name'], show_internal=True))
        logging.info(str(text + " newick " + tree.write(format=1)))


def add_internal_names(original_tree):
    for i, node in enumerate(original_tree.traverse()):
        if not node.is_leaf():
            node.name = "N{}".format(i)
        original_tree.get_tree_root().name = "ROOT"
    return original_tree


def generate_tree_object_from_newick(tree_path):
    starting_tree_object = Tree(newick=tree_path, format=1)
    add_internal_names(starting_tree_object)
    starting_tree_object.get_tree_root().name = "ROOT"
    return starting_tree_object


def generate_multiple_tree_object_from_newick(trees_path):
    with open(trees_path) as trees_path:
        newicks = trees_path.read().split("\n")
        newicks = [t for t in newicks if len(t) > 0]
        tree_objects = [generate_tree_object_from_newick(newick) for newick in newicks]
        return tree_objects


def get_tree_string(tree_path):
    tree_object = Tree(newick=tree_path, format=1)
    return (tree_object.write(format=1))


def compute_tree_divergence(tree):
    total_dist = 0
    for node in tree.iter_descendants():
        # Do some analysis on node
        total_dist = total_dist + node.dist
    return total_dist


def assign_brlen_to_tree_object(tree_object, brlen_list):
    for i, node in enumerate(tree_object.iter_descendants()):
        # Do some analysis on node
        node.dist = brlen_list[i]
    return tree_object


def EVAL_tree_objects_ll(tree_objects, curr_run_directory, msa_path, msa_type, opt_brlen = False):
    tmp_folder = os.path.join(curr_run_directory,"ll_evaluation_on_trees")
    create_or_clean_dir(tmp_folder)
    trees_path = os.path.join(tmp_folder,"SPR_trees_evaluation")
    with open(trees_path, 'w') as BEST_TREE:
        for obj in tree_objects:
            newick = (obj.write(format=1))
            BEST_TREE.write(newick)
    trees_ll,tree_objects = raxml_optimize_trees_for_given_msa(msa_path, "trees_eval", trees_path,
                                       tmp_folder,  msa_type, opt_brlen=opt_brlen
                                       )
    return trees_ll



def compute_largest_branch_length(tree):
    return max([node.dist for node in tree.iter_descendants()])


def max_distance_between_leaves(tree):
    max_dist = -1
    for leaf_a in tree.iter_leaves():
        for leaf_b in tree.iter_leaves():
            dist = tree.get_distance(leaf_a, leaf_b)
            if dist> max_dist:
                max_dist = dist
    return max_dist


def mad_tree_parameter(tree_path):
        mad_command = "{mad_exe_path} -t -s {tree_path}".format(mad_exe_path=MAD_COMMAND_PREFIX,
                                                                tree_path=tree_path)
        execute_command_and_write_to_log(mad_command)
        mad_log_path = tree_path + ".rooted"
        mad = extract_mad_file_statistic(mad_log_path)
        return mad

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



def main():
    t = Tree('((((H,K)D,(F,I)G)B,E)A,((L,(N,Q)O)J,(P,S)M)C);', format=1)
    add_internal_names(t)
    (print(t.get_ascii(attributes=['name'], show_internal=True)))
    # for i, pruning_head_node in enumerate(t.iter_descendants("levelorder")):
    #     if not pruning_head_node.up.is_root(): # if this is not one of the two direct child nodes of the root
    #         pruning_edge = Edge(node_a=pruning_head_node.name, node_b=pruning_head_node.up.name)
    #     for j, regrafting_head_node in enumerate(t.iter_descendants("levelorder")):
    #         if not regrafting_head_node.up.is_root():
    #             regrafting_edge = Edge(node_a=regrafting_head_node.name, node_b=regrafting_head_node.up.name)
    #             if not ((pruning_edge.node_a == regrafting_edge.node_a) or (pruning_edge.node_b == regrafting_edge.node_b) or (
    #                     pruning_edge.node_b == regrafting_edge.node_a) or (pruning_edge.node_a == regrafting_edge.node_b))



if __name__ == "__main__":
    main()
