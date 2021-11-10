from ete3 import *
import logging
import re

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


# def get_list_of_edges(starting_tree):
#     edges_list = []
#     main_tree_root_pointer_cp = starting_tree.copy()
#     for i, prune_node in enumerate(main_tree_root_pointer_cp.iter_descendants("levelorder")):
#         if prune_node.up:
#             edge = Edge(node_a=prune_node.name, node_b=prune_node.up.name)
#             edges_list.append(edge)
#     return edges_list


def get_distance_between_edges(tree, pruned_edge,regraft_edge):
    if tree.get_common_ancestor(regraft_edge.node_a, pruned_edge.node_a).name == (tree&pruned_edge.node_a).name:
        dist = (tree & pruned_edge.node_a).get_distance((tree & regraft_edge.node_a),topology_only=True)
    else:
        dist = (tree & pruned_edge.node_b).get_distance((tree & regraft_edge.node_a), topology_only=True)
    return dist


def get_possible_spr_moves(starting_tree, rearr_dist=-1):
    edges_list = []
    main_tree_root_pointer_cp = starting_tree.copy()
    for i, prune_node in enumerate(main_tree_root_pointer_cp.iter_descendants("levelorder")):
        if prune_node.up:
            edge = Edge(node_a=prune_node.name, node_b=prune_node.up.name)
            edges_list.append(edge)
    possible_moves = []
    for prune_edge in edges_list:
        for rgft_edge in edges_list:
            curr_rearr_dist = get_distance_between_edges(starting_tree,prune_edge,rgft_edge)
            if rearr_dist<0 or curr_rearr_dist<=rearr_dist:
                if not ((prune_edge.node_a == rgft_edge.node_a) or (prune_edge.node_b == rgft_edge.node_b) or (
                        prune_edge.node_b == rgft_edge.node_a) or (prune_edge.node_a == rgft_edge.node_b)):
                    possible_moves.append((prune_edge, rgft_edge))
    return possible_moves


def add_subtree_to_basetree(subtree_root, basetree_root, regraft_edge, length_regraft_edge, length_pruned_edge):
    future_sister_tree_to_pruned_tree = (basetree_root & regraft_edge.node_a).detach()
    new_tree_adding_pruned_and_future_sister = Tree()
    new_tree_adding_pruned_and_future_sister.add_child(subtree_root.copy(),
                                                       dist=length_pruned_edge)
    new_tree_adding_pruned_and_future_sister.add_child(future_sister_tree_to_pruned_tree, dist=length_regraft_edge / 2)
    (basetree_root & regraft_edge.node_b).add_child(new_tree_adding_pruned_and_future_sister,
                                                    dist=length_regraft_edge / 2)
    basetree_root.unroot()
    return basetree_root


def generate_neighbour(base_tree, possible_move):
    base_tree = base_tree.copy()  # not working on original tree
    pruned_edge, regraft_edge = possible_move
    length_regraft_edge = (base_tree & regraft_edge.node_a).dist
    length_pruned_edge = (base_tree & pruned_edge.node_a).dist
    if base_tree.get_common_ancestor(regraft_edge.node_a, pruned_edge.node_a).name == pruned_edge.node_a:
        new_base_tree = (base_tree & pruned_edge.node_a).detach()
        new_subtree_to_be_regrafted = base_tree
        if not (
                       new_subtree_to_be_regrafted & pruned_edge.node_b).name == new_subtree_to_be_regrafted.get_tree_root().name:
            new_subtree_to_be_regrafted.set_outgroup(new_subtree_to_be_regrafted & pruned_edge.node_b)
        (new_subtree_to_be_regrafted & pruned_edge.node_b).delete(preserve_branch_length=True)
        output_tree = add_subtree_to_basetree(new_subtree_to_be_regrafted, new_base_tree, regraft_edge,
                                              length_regraft_edge, length_pruned_edge)
    else:
        pruned_subtree = (base_tree & pruned_edge.node_a).detach()
        (base_tree & pruned_edge.node_b).delete(preserve_branch_length=True)
        output_tree = add_subtree_to_basetree(pruned_subtree, base_tree, regraft_edge, length_regraft_edge,
                                              length_pruned_edge)
    return output_tree


def add_internal_names(original_tree):
    for i, node in enumerate(original_tree.traverse()):
        if not node.is_leaf():
            node.name = "N{}".format(i)
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
        return tree_objects if len(tree_objects) > 1 else tree_objects[0]


def get_tree_string(tree_path):
    tree_object = Tree(newick=tree_path, format=1)
    return (tree_object.write(format=1))


def compute_tree_divergence(tree_path):
    total_dist = 0
    tree = generate_tree_object_from_newick(tree_path)
    for node in tree.iter_descendants():
        # Do some analysis on node
        total_dist = total_dist + node.dist
    return total_dist


def assign_brlen_to_tree_object(tree_object, brlen_list):
    for i, node in enumerate(tree_object.iter_descendants()):
        # Do some analysis on node
        node.dist = brlen_list[i]
    return tree_object



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

def compute_tree_divergence(tree_path):
    total_dist = 0
    tree = generate_tree_object_from_newick(tree_path)
    for node in tree.iter_descendants():
        # Do some analysis on node
        total_dist = total_dist + node.dist
    return total_dist



