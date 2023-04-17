import sys

if sys.platform == "linux" or sys.platform == "linux2":
    PROJECT_ROOT_DIRECRTORY = "/groups/pupko/noaeker/RAxML_params_tuning"
else:
    PROJECT_ROOT_DIRECRTORY = "/Users/noa/Workspace/RAxML_params_tuning"
sys.path.append(PROJECT_ROOT_DIRECRTORY)

from side_code.config import *
from side_code.code_submission import execute_command_and_write_to_log
from side_code.file_handling import create_dir_if_not_exists
from side_code.basic_trees_manipulation import generate_multiple_tree_object_from_newick_file
from side_code.raxml import raxml_optimize_trees_for_given_msa
import random
import shutil
import argparse
import os
import pandas as pd


def simulate_msa(output_prefix, tree_file, model, length, r_i, r_d, seed):
    '''

    :param output_prefix:
    :param model: WAG / GTR
    :param treefile:
    :return:
    '''
    command = f"{IQTREE_SIM_PATH} --alisim {output_prefix} -m {model} -t {tree_file}  --length {length}" \
        f" -seed {seed}  --indel {r_i},{r_d}"
    print(command)
    execute_command_and_write_to_log(command)
    msa_path = f"{output_prefix}.phy"
    return msa_path, tree_file

def extract_tree_files_from_folder(general_dir_path):
        files_list = []
        if os.path.exists(general_dir_path):
            for sub_dir in os.listdir(general_dir_path):
                sub_dir_path = os.path.join(general_dir_path, sub_dir)
                if os.path.isdir(sub_dir_path):
                    for file in os.listdir(sub_dir_path):
                        if file.endswith('.newick'):
                            files_list.append(os.path.join(sub_dir_path, file))
        return files_list

def simulate_MSAs(curr_run_directory, trees_folder, model, df):
    MSAs_dir = os.path.join(curr_run_directory, f"simulated_MSAs_{model}")
    MSAs_only_dir = os.path.join(curr_run_directory, f"simulated_MSAs_only_{model}")
    create_dir_if_not_exists(MSAs_dir)
    create_dir_if_not_exists(MSAs_only_dir)
    trees = extract_tree_files_from_folder(trees_folder)
    for i, tree_path in enumerate(trees):
        if i==95:
            continue
        r_i = random.uniform(0.01, 0.05)
        r_d = random.uniform(0.01, 0.05)
        msa_length = random.randint(100, 5000)
        output_prefix = os.path.join(MSAs_dir, f'sim_MSA_{i}_{model}')
        msa_path, tree_path = simulate_msa(output_prefix, tree_path, model, msa_length, r_i, r_d, seed=i)
        shutil.copy(msa_path, MSAs_only_dir)
        with open(tree_path) as TREE:
            tree_str = TREE.read()
        df = df.append({'tree_str': tree_str, 'msa_local_path': msa_path,  'tree_path': tree_path, 'model': model},
                       ignore_index=True)
    return df


def main():
    MSAs_folder = '/Users/noa/Workspace/data/New_MSAs'
    wag_trees_folder = f"{MSAs_folder}/out_WAG"
    create_dir_if_not_exists(wag_trees_folder)
    gtr_trees_folder = f"{MSAs_folder}/out_GTR"
    create_dir_if_not_exists(gtr_trees_folder)

    cmd = f'/Users/noa/Workspace/RAxMLGroveScripts/org_script.py -o {gtr_trees_folder} find -q  '+'"NUM_TAXA > 400 and NUM_TAXA<1000'+" and MODEL = 'GTR'"+'"'
    print(cmd)
    cmd = f'/Users/noa/Workspace/RAxMLGroveScripts/org_script.py -o {wag_trees_folder} find -q  ' + '"NUM_TAXA > 400 and NUM_TAXA<1000' + " and MODEL = 'WAG'" + '"'
    print(cmd)
    #parser = argparse.ArgumentParser()
    #parser.add_argument('--wag_folder', action='store', type=str, default= "/Users/noa/Workspace/data/New_MSAs/out_WAG")
    #args = parser.parse_args()
    #df_wag = simulate_MSAs(MSAs_folder, trees_folder= wag_trees_folder, model = 'WAG+G')
    csv_path = os.path.join(MSAs_folder, "simulated_MSAs_GTR.tsv")
    if os.path.exists(csv_path):
        df_gtr = pd.read_csv(csv_path, sep="#")
    else:
        df_gtr = pd.DataFrame()
    df_gtr = simulate_MSAs(MSAs_folder, trees_folder=gtr_trees_folder, model='GTR+G', df = df_gtr)
    #df = pd.concat([df_wag,df_gtr])
    df_gtr.to_csv(os.path.join(MSAs_folder, csv_path), sep='#', index = False, index_label= None)
    df_gtr = pd.read_csv(csv_path, sep= '#')
    pass


if __name__ == "__main__":
    main()
