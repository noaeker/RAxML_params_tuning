from ete3 import *
import logging
import numpy as np
from basic_trees_manipulation import *
from config import *


def get_distance_between_edges(tree, pruned_edge,regraft_edge):
    if tree.get_common_ancestor(regraft_edge.node_a, pruned_edge.node_a).name == (tree&pruned_edge.node_a).name:
        dist = (tree & pruned_edge.node_a).get_distance((tree & regraft_edge.node_a),topology_only=True)
    else:
        dist = (tree & pruned_edge.node_b).get_distance((tree & regraft_edge.node_a), topology_only=True)
    return dist

def get_all_possible_spr_moves(starting_tree, min_rearr_dist = -1, max_rearr_dist = np.inf):
    edges_list = []
    main_tree_root_pointer_cp = starting_tree.copy()
    for i, node in enumerate(main_tree_root_pointer_cp.iter_descendants("levelorder")):
        if node.up:
            edge = Edge(node_a=node.name, node_b=node.up.name)
            edges_list.append(edge)
    possible_moves = []
    for prune_edge in edges_list:
        for rgft_edge in edges_list:
            curr_rearr_dist = get_distance_between_edges(starting_tree,prune_edge,rgft_edge)
            if curr_rearr_dist>=min_rearr_dist and curr_rearr_dist<=max_rearr_dist:
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




def get_random_spr_moves_vs_distances(starting_tree_path, n_iterations,curr_run_directory, msa_path, msa_type):
    starting_tree = generate_tree_object_from_newick(starting_tree_path)
    edges_list = []
    main_tree_root_pointer_cp = starting_tree.copy()
    for i, node in enumerate(main_tree_root_pointer_cp.iter_descendants("levelorder")):
        if node.up:
            edge = Edge(node_a=node.name, node_b=node.up.name)
            edges_list.append(edge)
    distances = []
    ll_improvements = []
    starting_tree_ll = EVAL_tree_objects_ll([starting_tree], curr_run_directory, msa_path, msa_type)
    i=0
    while len(distances)<n_iterations:
            np.random.seed(SEED+i)
            i+=1
            prune_edge, rgft_edge = np.random.choice(edges_list, size=2, replace= False)
            curr_rearr_dist = get_distance_between_edges(starting_tree,prune_edge,rgft_edge)
            move=(prune_edge, rgft_edge)
            if not ((prune_edge.node_a == rgft_edge.node_a) or (prune_edge.node_b == rgft_edge.node_b) or (
                    prune_edge.node_b == rgft_edge.node_a) or (prune_edge.node_a == rgft_edge.node_b)):
                neighbour = generate_neighbour(starting_tree,move)
                neighbour_ll = EVAL_tree_objects_ll([neighbour], curr_run_directory, msa_path, msa_type)
                ll_improvements.append(neighbour_ll-starting_tree_ll)
                distances.append(curr_rearr_dist)
    print(distances)
    print(ll_improvements)
    return distances, ll_improvements




