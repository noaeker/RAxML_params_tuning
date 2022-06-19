import ModelTeller.msa_functions as msa_functions
import ModelTeller.tree_functions as tree_functions
from Bio import AlignIO
import ete3
import pandas as pd
import numpy as np


def break_model_column(df, column_name):
	df[column_name + "_I"] = 0
	df[column_name + "_G"] = 0
	df[column_name + "_F"] = 0
	df.loc[df[column_name].str.endswith("+I+G"), column_name + "_I"] = 1
	df.loc[df[column_name].str.endswith("+I"), column_name + "_I"] = 1

	df.loc[df[column_name].str.endswith("+I+G"), column_name + "_G"] = 1
	df.loc[df[column_name].str.endswith("+G"), column_name + "_G"] = 1

	df[column_name + "_model"] = df[column_name].apply(lambda x: x[:x.index("+")] if "+" in x else x)
	df.loc[df[column_name + "_model"].isin(["F81", "HKY", "GTR"]), column_name + "_F"] = 1

	df[column_name + "_matrix"] = 0
	df.loc[df[column_name + "_model"].isin(["K80", "HKY"]), column_name + "_matrix"] = 1
	df.loc[df[column_name + "_model"].isin(["SYM", "GTR"]), column_name + "_matrix"] = 2


def compute_tree_features(phyml_tree_filepath, feat_prefix):
	tree = tree_functions.get_newick_tree(str(phyml_tree_filepath))
	bl_estimates = tree_functions.get_branch_lengths_estimates(tree)
	tree_diam_estimates = tree_functions.get_diameters_estimates(str(phyml_tree_filepath))
	cnt_diam_estimates = tree_functions.get_diameters_estimates(str(phyml_tree_filepath), actual_bl=False)
	frac_cherries = tree_functions.get_frac_of_cherries(tree)
	largest_branch_node = tree_functions.get_largest_branch(tree)
	try:
		tree.set_outgroup(largest_branch_node)
	except ete3.coretype.tree.TreeError:
		pass

	stem85, stem90 = tree_functions.get_stemminess_indexes(tree)
	# stats_dict = phyml.parse_phyml_stats_file(phyml_stats_filepath)

	model_phyml_features_dict = {}
	model_phyml_features_dict["max_bl"], model_phyml_features_dict["min_bl"], \
	model_phyml_features_dict["mean_bl"], model_phyml_features_dict["std_bl"], \
	model_phyml_features_dict["entropy_bl"] = bl_estimates

	model_phyml_features_dict["max_diam"], model_phyml_features_dict["min_diam"], \
	model_phyml_features_dict["mean_diam"], model_phyml_features_dict["std_diam"], \
	model_phyml_features_dict["entropy_diam"] = tree_diam_estimates

	model_phyml_features_dict["max_diam_cnt"], model_phyml_features_dict["min_diam_cnt"], \
	model_phyml_features_dict["mean_diam_cnt"], model_phyml_features_dict["std_diam_cnt"], \
	model_phyml_features_dict["entropy_diam_cnt"] = cnt_diam_estimates

	model_phyml_features_dict["frac_cherries"] = frac_cherries
	model_phyml_features_dict["stemminess85_idx"], model_phyml_features_dict["stemminess90_idx"] = stem85, stem90

	# model_phyml_features_dict.update(stats_dict)
	# model_phyml_features_dict.pop("Tstv")

	new_dict = {}
	for k in model_phyml_features_dict:
		if not model_phyml_features_dict[k] is None:
			new_dict[feat_prefix + k] = model_phyml_features_dict[k]
	return tree, new_dict


def calculate_alignment_features(msa, reduced=False):
	pinv_100 = msa_functions.count_fully_conserved_fraction(msa)
	entropy = msa_functions.get_msa_avg_entropy(msa)
	bb_multinomial, n_unique_sites, frac_unique_sites = msa_functions.calculate_bollback_multinomial(msa)

	sample = {}
	sample["pinv_sites_100p"] = pinv_100
	sample["aln_entropy"] = entropy
	sample["bollback_multinomial"], sample["n_unique_sites"], sample["frac_unique_sites"] = \
		bb_multinomial, n_unique_sites, frac_unique_sites

	if not reduced:
		freqs = msa_functions.compute_base_frequencies(msa)
		substitution_statistics_dict, pairiwse_substitution_values_dict \
			= msa_functions.calculate_substitution_rates(msa)

		sample.update(substitution_statistics_dict)
		sample.update(pairiwse_substitution_values_dict)
		sample.update(freqs)
	else:
		new_dict = {}
		for k in sample:
			new_dict["rmsa_" + k] = sample[k]
		sample = new_dict

	return sample


def extract_features(msa_file, user_tree_file):
	"""
	:param msa: path to msa file
	:param msa_file: path to msa file
	:param GTRIG_topology: false
	:param user_tree_file: tree file
	:return:
	"""
	msa = AlignIO.read(msa_file, 'fasta')
	# extract from MSA
	ntaxa, nchars = msa_functions.get_msa_properties(msa)
	msa_features_dict = calculate_alignment_features(msa)

	opt_rates_model = ""
	a_tree, tree_features_dict = compute_tree_features(user_tree_file, opt_rates_model)
	# compute MSA features for sequences without "outgroup" (set according to largest branch)
	outgroup_leaves, ingroup_leaves = \
		tree_functions.get_internal_and_external_leaves_relative_to_subroot \
			(a_tree, tree_functions.get_largest_branch(a_tree))
	if len(outgroup_leaves) > len(ingroup_leaves):
		ingroup_leaves, outgroup_leaves = outgroup_leaves, ingroup_leaves
	ingroup_names = [leaf.name for leaf in ingroup_leaves]
	reduced_msa = msa_functions.reduce_msa_to_seqs_by_name(msa, ingroup_names)

	rmsa_features_dict = calculate_alignment_features(reduced_msa, reduced=True)

	sample = {}
	sample["ntaxa"], sample["nchars"] = ntaxa, nchars
	sample.update(msa_features_dict)
	sample.update(rmsa_features_dict)
	sample.update(tree_features_dict)

	return sample


def prepare_features_df(msa_filepath, user_tree_file):
	all_features = extract_features(msa_filepath, user_tree_file)
	all_features["base_freqs_entropy"] = (np.log2(all_features[["freq_A", "freq_C", "freq_G", "freq_T"]]) *
										  all_features[["freq_A", "freq_C", "freq_G", "freq_T"]]).sum(axis=1) * -1

	return all_features
