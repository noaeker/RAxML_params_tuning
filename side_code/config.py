from sys import platform
import logging

if platform == "linux" or platform == "linux2":
    LOCAL_RUN = False
else:
    LOCAL_RUN = True

MODULE_LOAD_STR = "source /groups/itay_mayrose/lab_python/anaconda3/etc/profile.d/conda.sh; source activate noa_env;  module load gcc/gcc-7.2.0; module load R/3.6.1;"
PBS_FILE_GENERATOR_CODE = "/bioseq/bioSequence_scripts_and_constants/q_submitter_power.py"
POSSIBLE_PROTEIN_LETTERS = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T',
                            'W', 'Y', 'V']
SEED = 1
BASELINE = "X"
CURR_RUN_PREFIX = "pandit_tuning"
CURR_JOBS_PREFIX = "pandit_tuning"
N_CPUS_PER_JOB = 1
N_CPUS_PER_TRAINING = 1
N_CPUS_RAXML = 1
N_MSAS =5
N_JOBS = 2
N_TASKS_PER_JOB = 300
MIN_N_SEQ = 1
MAX_N_SEQ = 1000
N_PARSIMONY_RAXML_SEARCH = 40#15
N_RANDOM_RAXML_SEARCH = 40#15
MIN_N_LOCI = 1
LOGGING_LEVEL = "info" #"debug"
OUTPUT_CSV_NAME = "tune_raxml"
WAITING_TIME_UPDATE = 20
TEST_MSA_ITERATIONS = 30
EPSILON = 0.1
SPR_RADIUS_GRID =  "1"#"1_30_10"
SPR_CUTOFF_GRID = "0.5"#"0.1_10_10"
CSV_SEP = "\t"
CSV_SUFFIX = ".tsv"
N_MSAS_PER_BUNCH = -1
MSAs_POOL_SIZE = 1000




# PATH CONFIGURATION

if not LOCAL_RUN:
    IQTREE_EXE = "/groups/pupko/noaeker/programs/tree_search_programs/iqtree/bin/iqtree"
    RAXML_NG_EXE = "/groups/pupko/noaeker/programs/tree_search_programs/raxml-ng/raxml-ng"
    MAD_COMMAND_PREFIX = "/groups/pupko/noaeker/programs/other_programs/mad"
    RESULTS_FOLDER = "/groups/pupko/noaeker/RAxML_params_tuning_results"
    GENERAL_MSA_DIR = "/groups/pupko/noaeker/data/new_MSAs"
    PROJECT_ROOT_DIRECRTORY = "/groups/pupko/noaeker/RAxML_params_tuning"
    MAIN_CODE_PATH = f"raw_data_generation/tune_params.py"
    SAMPLING_MAIN_CODE_PATH = f"raw_data_sampling_analysis/random_and_parsimony_sampling.py"
    FEATURE_EXTRACTION_CODE = f"{PROJECT_ROOT_DIRECRTORY}/feature_extraction/features_extractor_new_version.py"
    GROUPS_FEATURE_EXTRACTION_CODE = f"{PROJECT_ROOT_DIRECRTORY}/groups_paper_ML_code/groups_data_generation.py"
    RAXML_HPC_EXE = "/groups/pupko/noaeker/standard-RAxML/raxmlHPC"
    RATE4SITE_COMMAND_PREFIX = "/groups/pupko/noaeker/programs/other_programs/rate4site/rate4site"
    TMP_RESULTS_FOLDER = f"{RESULTS_FOLDER}/single_gene_MSAs"
    CSV_MSAs_FOLDER = "/groups/pupko/noaeker/data/ABC_DR"
    READY_RAW_DATA = f"{RESULTS_FOLDER}/global_shared_results_c_30_70"

elif LOCAL_RUN:
    IQTREE_EXE = "/Users/noa/Programs/iqtree-2.1.3-MacOSX/bin/iqtree2"
    RAXML_NG_EXE = "/Users/noa/Programs/Raxml/raxml-ng"
    RAXML_HPC_EXE = "/Users/noa/Programs/standard-RAxML/raxmlHPC-PTHREADS "
    MAD_COMMAND_PREFIX = "/Users/noa/Programs/mad.osx"
    RATE4SITE_COMMAND_PREFIX = "/Users/noa/Programs/rate4site"
    RESULTS_FOLDER = "/Users/noa/Workspace/raxml_deep_learning_results"
    ML_RESULTS_FOLDER = f"{RESULTS_FOLDER}/ML_pipeline_results"
    LOCAL_DATA_GENERATION = f"{RESULTS_FOLDER}/local_data_generation"
    READY_RAW_DATA = f"{RESULTS_FOLDER}/ready_raw_data"
    MSAs_CSV_PATH = "/Users/noa/Workspace/data/sampled_datasets.csv"
    GENERAL_MSA_DIR = "/Users/noa/Workspace/data/New_MSAs/Single_gene_DNA"#'/Users/noa/Workspace/data/new_MSAs/Selectome_msas'#"/Users/noa/Workspace/data/single-gene_alignments"
    PROJECT_ROOT_DIRECRTORY = "/Users/noa/Workspace/RAxML_params_tuning"
    MAIN_CODE_PATH = f"{PROJECT_ROOT_DIRECRTORY}/raw_data_generation/tune_params.py"
    SAMPLING_MAIN_CODE_PATH = f"{PROJECT_ROOT_DIRECRTORY}/raw_data_sampling_analysis.py"
    FEATURE_EXTRACTION_CODE = f"{PROJECT_ROOT_DIRECRTORY}/feature_extraction/features_extractor_new_version.py"
    GROUPS_FEATURE_EXTRACTION_CODE = f"{PROJECT_ROOT_DIRECRTORY}/groups_paper_ML_code/groups_data_generation.py"
    TMP_RESULTS_FOLDER =  f"{RESULTS_FOLDER}/single_gene_MSAs"
    CSV_MSAs_FOLDER = "/Users/noa/Workspace/data/ABC_DR"
