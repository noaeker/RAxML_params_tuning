from ete3 import *
from help_functions import *



def print_subtree(tree, log_file, text):
    if log_file:
        log_file.write(text + " visualization: " + "\n" + tree.get_ascii(attributes=['name'],
                                                                         show_internal=True) + "\n")
        log_file.write(text + " newick " + str(tree.write(format=1)) + "\n")
    else:
        logging.info(text + " visualization: " + "\n" + tree.get_ascii(attributes=['name'], show_internal=True))
        logging.info(str(text + " newick " + tree.write(format=1)))


def generate_tree_object_from_newick(tree_path):
    starting_tree_object = Tree(newick=tree_path, format=1)
    add_internal_names(starting_tree_object)
    starting_tree_object.get_tree_root().name = "ROOT"
    return starting_tree_object

def add_internal_names(original_tree):
    for i, node in enumerate(original_tree.traverse()):
        if not node.is_leaf():
            node.name = "N{}".format(i)
        original_tree.get_tree_root().name = "ROOT"
    return original_tree


def get_tree_string(tree_path):
    tree_object = Tree(newick=tree_path, format=1)
    return (tree_object.write(format=1))


def compute_tree_divergence(tree):
    total_dist = 0
    for node in tree.iter_descendants():
        # Do some analysis on node
        total_dist = total_dist + node.dist
    return total_dist

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
        execute_commnand_and_write_to_log(mad_command)
        mad_log_path = tree_path + ".rooted"
        mad = extract_mad_file_statistic(mad_log_path)
        return mad





if __name__ == "__main__":
    main()