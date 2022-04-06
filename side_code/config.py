from sys import platform
import logging

if platform == "linux" or platform == "linux2":
    LOCAL_RUN = False
else:
    LOCAL_RUN = True

MODULE_LOAD_STR = "source /groups/itay_mayrose/lab_python/anaconda3/etc/profile.d/conda.sh; source activate noa_env;  module load gcc/gcc-7.2.0; module load R/3.6.1;"
PBS_FILE_GENERATOR_CODE = "/bioseq/bioSequence_scripts_and_constants/q_submitter_power.py"
SEED = 1
BASELINE = "X"
CURR_RUN_PREFIX = "first_tuning"
CURR_JOBS_PREFIX = "first_tuning"
N_CPUS_PER_JOB = 1
N_CPUS_PER_TRAINING = 1
N_CPUS_RAXML = 1
N_MSAS = 2
N_JOBS = 2
N_TASKS_PER_JOB = 5
MIN_N_SEQ = 10
N_PARSIMONY_RAXML_SEARCH = 0#15
N_RANDOM_RAXML_SEARCH = 1#15
MIN_N_LOCI = 1
LOGGING_LEVEL = logging.INFO
OUTPUT_CSV_NAME = "tune_raxml"
WAITING_TIME_UPDATE = 20
TEST_MSA_ITERATIONS = 10
EPSILON = 0.1
SPR_RADIUS_GRID =  "1_30_1"#"1_30_10"
SPR_CUTOFF_GRID = "1_30_1"#"0.1_10_10"
CSV_SEP = "\t"
CSV_SUFFIX = ".tsv"

COLUMNS_TO_INCLUDE_CSV = ["msa_folder","original_alignment_path","msa_name","elapsed_running_time", "msa_type", "n_loci", "n_seq", "run_name",
                         "spr_cutoff","spr_radius","starting_tree_ll","final_ll","rf_from_curr_starting_tree_best_topology","curr_starting_tree_best_ll",
                         "delta_ll_from_curr_starting_tree_best_topology","starting_tree_ind","tree_type","best_msa_ll","rf_from_overall_msa_best_topology",
                         "delta_ll_from_overall_msa_best_topology"]



# PATH CONFIGURATION

if not LOCAL_RUN:
    RAXML_NG_EXE = "/groups/pupko/noaeker/programs/tree_search_programs/raxml-ng/raxml-ng  "
    MAD_COMMAND_PREFIX = "/groups/pupko/noaeker/mad"
    RESULTS_FOLDER = "/groups/pupko/noaeker/RAxML_params_tuning_results"
    GENERAL_MSA_DIR = "/groups/pupko/noaeker/data/single-gene_alignments"
    MAIN_CODE_PATH = "/groups/pupko/noaeker/RAxML_params_tuning/raw_data_generation/tune_params.py"
    SAMPLING_MAIN_CODE_PATH = "/groups/pupko/noaeker/RAxML_params_tuning//raw_data_generation/random_and_parsimony_sampling.py"
    RAXML_HPC_EXE = "/groups/pupko/noaeker/standard-RAxML/raxmlHPC"
    RATE4SITE_COMMAND_PREFIX = "/groups/pupko/noaeker/programs/other_programs/rate4site/rate4site"
    TMP_RESULTS_FOLDER = f"{RESULTS_FOLDER}/single_gene_MSAs"

elif LOCAL_RUN:
    IQTREE_EXE = "/Users/noa/Programs/iqtree-2.1.3-MacOSX/bin/iqtree2"
    RAXML_NG_EXE = "/Users/noa/Programs/Raxml/raxml-ng  "
    RAXML_HPC_EXE = "/Users/noa/Programs/standard-RAxML/raxmlHPC-PTHREADS "
    MAD_COMMAND_PREFIX = "/Users/noa/Programs/mad.osx"
    RATE4SITE_COMMAND_PREFIX = "/Users/noa/Programs/rate4site"
    RESULTS_FOLDER = "/Users/noa/Workspace/raxml_deep_learning_results"
    MSAs_CSV_PATH = "/Users/noa/Workspace/data/sampled_datasets.csv"
    GENERAL_MSA_DIR = "/Users/noa/Workspace/data/single-gene_alignments"
    MAIN_CODE_PATH = "/Users/noa/Workspace/RAxML_params_tuning/raw_data_generation/tune_params.py"
    SAMPLING_MAIN_CODE_PATH = "/Users/noa/Workspace/RAxML_params_tuning/raw_data_generation/random_and_parsimony_sampling.py"
    TMP_RESULTS_FOLDER =  f"{RESULTS_FOLDER}/single_gene_MSAs"

