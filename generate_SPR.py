from raxml import *
import numpy as np
from scipy import stats
from trees_manipulation import *
from sklearn.metrics import *


def write_spr_log_message(spr_log_file_object, rgrft_path, best_ll, ll, best_topology_path):
    if ll > best_ll:
        spr_log_file_object.write("Found a better tree!!!! in " + rgrft_path + "\n")
        spr_log_file_object.write("    1. best ll is: " + str(best_ll) + "\n")
        spr_log_file_object.write("    2. current tree in " + rgrft_path + " has log likelihood of " + str(ll) + "\n")
        spr_log_file_object.write("    copying topology in " + rgrft_path + " to " + best_topology_path + "\n")
        spr_log_file_object.write("updating best likelihood to be " + str(ll) + "\n")
    elif ll > -np.infty:
        spr_log_file_object.write("Not found a better tree in " + rgrft_path + "\n")
        spr_log_file_object.write("    1. best ll is: " + str(best_ll) + "\n")
        spr_log_file_object.write("    2. current tree in " + rgrft_path + " has log likelihood of " + str(ll) + "\n")


def get_true_ll_values(curr_msa_stats, trees_path, curr_run_directory,opt_brlen):
        trees_true_ll, trees_true_optimized_objects, time_rgft_eval_true = raxml_optimize_trees_for_given_msa(
                curr_msa_stats["local_alignment_path"],
                "rgrft_ll_eval_on_full_MSA", trees_path,
                curr_msa_stats, curr_run_directory,
                weights=None, n_cpus=curr_msa_stats["n_cpus_full"], opt_brlen= opt_brlen)
        return trees_true_ll



def compute_true_ll_of_best_tree_of_spr_iteration(weights_file_path, best_tree_object, curr_run_directory, curr_msa_stats,
                                                  trees_true_ll, best_ll_index, best_ll, top_x_to_test):
    if weights_file_path and top_x_to_test == 1:
        if curr_msa_stats["compute_all_true_ll"]:
            return trees_true_ll[best_ll_index]
        else:
            best_first_phase_newick = (best_tree_object.write(format=1))
            best_tree_path = os.path.join(curr_run_directory, "iteration_best_spr_tree")
            with open(best_tree_path, 'w') as BEST_TREE:
                BEST_TREE.write(best_first_phase_newick)
            best_tree_true_ll, best_tree_true_optimized_object, best_tree_true_eval_true = raxml_optimize_trees_for_given_msa(
                curr_msa_stats["local_alignment_path"],
                "best_iter_tree_eval_full_MSA", best_tree_path,
                curr_msa_stats, curr_run_directory,
                weights=None, n_cpus=curr_msa_stats["n_cpus_full"])
            best_true_ll = best_tree_true_ll
            return best_true_ll
    else:
        return best_ll


def regression_correct_lasso_ll_values(lasso_intercept,weights_file_path,trees_ll):
    if weights_file_path:
        ll_fixed = [((ll) + lasso_intercept) / INTEGER_CONST for ll in trees_ll]
        return ll_fixed
    else:
        return trees_ll




def write_tree_objects_to_file(trees_objects,curr_run_directory, top_trees_file_name):
    top_ll_tree_objects = np.array(trees_objects)
    top_ll_trees_newick = "\n".join([tree_object.write(format=1) for tree_object in top_ll_tree_objects])
    top_ll_trees_path = os.path.join(curr_run_directory, top_trees_file_name)
    with open(top_ll_trees_path, 'w') as TOP_LL_TREES:
        TOP_LL_TREES.write(top_ll_trees_newick)
    return top_ll_trees_path





def get_non_greedy_optimized_SPR_neighbours(curr_msa_stats, MSA_path, unique_trees_path,curr_run_directory, n_cpus):
    if curr_msa_stats["optimized_neighbours_per_iter"] > 1:
        logging.info("Evaluating (no brlen opt) LL of all SPR neighbours")
        trees_ll_no_brlen, trees_optimized_objects_no_brlen, time_rgft_eval_no_brlen = raxml_optimize_trees_for_given_msa(
            MSA_path, "rgrft_ll_eval_no_brlen",
            unique_trees_path,
            curr_msa_stats,
            curr_run_directory, opt_brlen=False,
            n_cpus= n_cpus)
        indices_of_spr_candidates_for_brlen_opt = (-np.array(trees_ll_no_brlen)).argsort()[
                                                  :curr_msa_stats["optimized_neighbours_per_iter"]]
        tree_objects_of_spr_candidates_for_brlen_opt = np.array(trees_optimized_objects_no_brlen)[
            indices_of_spr_candidates_for_brlen_opt]
        spr_candidates_for_brlen_opt_file = write_tree_objects_to_file(tree_objects_of_spr_candidates_for_brlen_opt,
                                                                       curr_run_directory,
                                                                       "spr_candidates_for_brlen_opt.trees")
        logging.info("About to optimize LL of most promising {t} topologies".format(t=curr_msa_stats["optimized_neighbours_per_iter"]))
    else:
        logging.info("Fully optimizing all SPR neighbours")
        spr_candidates_for_brlen_opt_file = unique_trees_path
    ll_spr_candidates_for_brlen, optimized_objects_spr_candidates_for_brlen, time_rgft_eval_true_spr_candidates_for_brlen = raxml_optimize_trees_for_given_msa(
        MSA_path, "rgrft_ll_eval_brlen",
        spr_candidates_for_brlen_opt_file,
        curr_msa_stats,
        curr_run_directory,
        n_cpus=n_cpus)
    res = {"spr_candidates_for_brlen_opt_file": spr_candidates_for_brlen_opt_file,"trees_ll_no_brlen" : trees_ll_no_brlen, "ll_spr_candidates_for_brlen": ll_spr_candidates_for_brlen,
           "iteration_time_brlen": time_rgft_eval_true_spr_candidates_for_brlen, "iteration_time_no_brlen" :time_rgft_eval_no_brlen, "optimized_objects_spr_candidates_for_brlen": optimized_objects_spr_candidates_for_brlen
           }
    return res


def re_optimize_some_SPR_neighbours_no_weights(ll_spr_candidates_for_brlen_corrected,top_x_true_trees,optimized_objects_spr_candidates_for_brlen, curr_msa_stats, curr_run_directory):
    top_ll_indices = (-np.array(ll_spr_candidates_for_brlen_corrected)).argsort()[:top_x_true_trees]
    top_ll_tree_objects = np.array(optimized_objects_spr_candidates_for_brlen)[top_ll_indices]
    top_ll_trees_newick = "\n".join([tree_object.write(format=1) for tree_object in top_ll_tree_objects])
    top_ll_trees_path = os.path.join(curr_run_directory, "lasso_top_ll_trees_file.trees")
    with open(top_ll_trees_path, 'w') as TOP_LL_TREES:
        TOP_LL_TREES.write(top_ll_trees_newick)
    top_trees_true_ll, top_trees_true_optimized_objects, time_rgft_eval_true = raxml_optimize_trees_for_given_msa(
        curr_msa_stats["local_alignment_path"],
        "lasso_re_optimization_on_full_MSA", top_ll_trees_path,
        curr_msa_stats, curr_run_directory,
        weights=None, n_cpus= curr_msa_stats["n_cpus_full"])  # optimize without weights
    best_ll = max(top_trees_true_ll)
    best_ll_index = top_trees_true_ll.index(best_ll)
    best_tree_object = top_trees_true_optimized_objects[best_ll_index]
    return best_ll, best_ll_index, best_tree_object,time_rgft_eval_true



def SPR_iteration(MSA_path, curr_msa_stats, starting_tree_object,
                  curr_run_directory
                  , n_cpus):
    iteration_time = 0
    add_internal_names(starting_tree_object)
    starting_tree_object.get_tree_root().name = "ROOT"
    logging.debug(str(starting_tree_object.write(format=1)) + "\n")
    starting_tree_spr_neighbours = get_possible_spr_moves(starting_tree_object, rearr_dist = curr_msa_stats["rearr_dist"])
    all_radius_spr_neighbours = [generate_neighbour(starting_tree_object, spr_neighbour) for spr_neighbour in
                       starting_tree_spr_neighbours]
    regrafted_trees_newick = "\n".join([regrafted_tree.write(format=1) for regrafted_tree in all_radius_spr_neighbours])
    trees_eval_path = os.path.join(curr_run_directory, "iteration_spr_trees")
    with open(trees_eval_path, 'w') as EVAL_TREES:
        EVAL_TREES.write(regrafted_trees_newick)
    unique_spr_neighbours_path = filter_unique_topologies(curr_run_directory, trees_eval_path, len(all_radius_spr_neighbours))

    spr_evaluation_data = get_non_greedy_optimized_SPR_neighbours(curr_msa_stats, MSA_path, unique_spr_neighbours_path, curr_run_directory, n_cpus)

    iteration_time+= spr_evaluation_data["iteration_time_brlen"]+spr_evaluation_data["iteration_time_no_brlen"]
    best_ll, best_ll_index, best_tree_object, re_optimization_time = re_optimize_some_SPR_neighbours_no_weights(spr_evaluation_data["ll_spr_candidates_for_brlen_corrected"],spr_evaluation_data["optimized_objects_spr_candidates_for_brlen"], curr_msa_stats, curr_run_directory)
    iteration_time += re_optimization_time

    best_ll = max(spr_evaluation_data["ll_spr_candidates_for_brlen"])
    best_ll_index = spr_evaluation_data["ll_spr_candidates_for_brlen"].index(best_ll)
    best_tree_object = spr_evaluation_data["optimized_objects_spr_candidates_for_brlen"][best_ll_index]

    results_dict = {"best_tree_object":best_tree_object, "best_ll" : best_ll
                    , "iteration_time": iteration_time, "iteration_time_brlen" : spr_evaluation_data["iteration_time_brlen"], "iteration_time_no_brlen":spr_evaluation_data["iteration_time_no_brlen"], "n_neighbours": len(spr_evaluation_data["trees_ll_no_brlen"]) }
    return results_dict





def SPR_search(MSA_path, run_unique_name, curr_msa_stats, starting_tree_path,
               curr_run_directory
              , n_cpus = 1):
    running_times_per_iter = []
    no_brlen_times_per_iter = []
    brlen_per_iter = []
    re_optimization_time_per_iter = []
    spr_iterations_performed_so_far = 0
    total_spr_neighbours_evaluated=0
    search_starting_tree_ll, tree_objects, elapsed_running_time_starting_eval = raxml_optimize_trees_for_given_msa(
        MSA_path,
        "starting_tree_ll_eval_" + run_unique_name,
        starting_tree_path,
        curr_msa_stats,
        curr_run_directory=curr_run_directory,
        n_cpus=n_cpus)

    logging.info("Search starting tree ll = {}. Using only true starting tree values ".format(search_starting_tree_ll
                                                                                         ))

    LL_per_iteration_list = [search_starting_tree_ll]
    curr_best_tree_ll = search_starting_tree_ll
    starting_tree_object = generate_tree_object_from_newick(starting_tree_path)
    curr_best_tree_object = starting_tree_object
    while True:
        curr_iter_run_directory = os.path.join(curr_run_directory, "iter_" + str(spr_iterations_performed_so_far))
        create_or_clean_dir(curr_iter_run_directory)
        logging.debug("iteration number: " + str(spr_iterations_performed_so_far))
        new_iteration_results= SPR_iteration(
            spr_iterations_performed_so_far, MSA_path, curr_msa_stats, curr_best_tree_object,
            curr_iter_run_directory
            , n_cpus
        )
        logging.debug(
            "Our current best tree ll is {} (its true ll is {}), best neighbour ll is {}".format(curr_best_tree_ll,
                                                                                                 curr_best_tree_true_ll,
                                                                                                 new_iteration_results["best_ll"]))
        running_times_per_iter.append(new_iteration_results["iteration_time"])
        brlen_per_iter.append(new_iteration_results["iteration_time_brlen"])
        no_brlen_times_per_iter.append(new_iteration_results["iteration_time_no_brlen"])
        if new_iteration_results["best_ll"] - curr_best_tree_ll <= EPSILON:
            logging.debug(
                "Difference between best spr neighbour and current tree <= {}, stopping SPR search\n".format(EPSILON))
            break
        logging.debug("Updating best neighbour to be our current best tree! ")
        ### Updating current iteration results and preparing for next iteration:
        spr_iterations_performed_so_far = spr_iterations_performed_so_far + 1
        curr_best_tree_object = new_iteration_results["best_tree_object"]
        curr_best_tree_ll = new_iteration_results["best_ll"]
        curr_best_tree_true_ll = new_iteration_results["best_true_ll"]
        LL_per_iteration_list += [curr_best_tree_ll]
        total_spr_neighbours_evaluated = total_spr_neighbours_evaluated+new_iteration_results["n_neighbours"]
    curr_best_tree_path = os.path.join(curr_run_directory, "search_best_tree_path")
    with open(curr_best_tree_path,'w') as BEST_TREE:
        BEST_TREE.write(curr_best_tree_object.write(format=1))
    search_results = {
        "search_best_ll": curr_best_tree_ll,
        "search_starting_tree_ll":  search_starting_tree_ll,
        "search_best_true_ll": curr_best_tree_true_ll,
        "search_best_topology_newick": curr_best_tree_object.write(format=1),
        "search_starting_tree_newick": starting_tree_object.write(format=1),
        "ll_per_iteration_list": LL_per_iteration_list,
        "search_best_tree_object": curr_best_tree_object,
        "search_best_tree_path": curr_best_tree_path,
        "search_spr_moves": spr_iterations_performed_so_far,
        "running_time_per_iter": running_times_per_iter,
        "total_search_running_time": sum(running_times_per_iter),
        "total_brlen_time" : sum(brlen_per_iter) ,
        "total_no_brlen_time" : sum(no_brlen_times_per_iter),
        "total_reoptimization_time" : sum (re_optimization_time_per_iter),
        "total_spr_neighbours_evaluated": total_spr_neighbours_evaluated
    }

    return search_results



