import logging

LOCAL_RUN = False

MODULE_LOAD_STR = "module load gcc/gcc-8.2.0; module load R/3.6.1; module load python/python-anaconda3.6.5-orenavr2; module load intel/parallel_studio_xe_2020.4.omnipath;"
PBS_FILE_GENERATOR_CODE = "/bioseq/bioSequence_scripts_and_constants/q_submitter_power.py"
SEED = 1
BASELINE = "X"
CURR_RUN_PREFIX = "first_tuning"
CURR_JOBS_PREFIX = "first_tuning"
N_CPUS_PER_JOB = 1
N_CPUS_PER_TRAINING = 1
N_CPUS_RAXML = 1
N_MSAS = 2
N_JOBS = 1
N_SEQ = 10
MIN_N_SEQ = 10
N_PARSIMONY_RAXML_SEARCH = 2
N_RANDOM_RAXML_SEARCH = 2
N_LOCI = 100
MIN_N_LOCI = 1
LOGGING_LEVEL = logging.DEBUG
OUTPUT_CSV_NAME = "tune_raxml"
WAITING_TIME_UPDATE = 60
TRAINING_SIZE = 1000
EPSILON = 0.1
SPR_RADIUS_GRID = "1_30_2"
SPR_CUTOFF_GRID = "0.1_10_2"

COLUMNS_TO_IGNORE_CSV = ["run_raxml_commands_locally", "run_prefix", "remove_output_files", "queue", "n_jobs",
                         "n_cpus_raxml", "n_cpus_per_job", "n_MSAs", "curr_job_folder", "msa_name", "min_n_loci",
                         "min_n_seq",
                         "jobs_prefix", "job_ind", "final_tree_topology", "first_msa_ind", "jobs_prefix", "job_ind",
                         "n_raxml_parsimony_trees", "n_raxml_random_trees",
                         "queue", "remove_output_files", "trim_msa", "spr_radius_grid", "spr_cutoff_grid"]

# PATH CONFIGURATION

if not LOCAL_RUN:
    RAXML_NG_EXE = "/groups/pupko/noaeker/raxml-ng-float-mpi/raxml-ng --extra thread-pin "
    MAD_COMMAND_PREFIX = "/groups/pupko/noaeker/mad"
    RESULTS_FOLDER = "/groups/pupko/noaeker/RAxML_params_tuning_results"
    MSAs_CSV_PATH = "/groups/pupko/noaeker/data/sampled_datasets.csv"
    GENERAL_MSA_DIR = "/groups/pupko/noaeker/data/single-gene_alignments"
    MAIN_CODE_PATH = "/groups/pupko/noaeker/lasso_positions_sampling/parallel_code/MSA_positions_sampling.py"
    R_CODE_PATH = "/groups/pupko/noaeker/lasso_positions_sampling/R_code/lasso_glmnet.R"
    RAXML_HPC_EXE = "/groups/pupko/noaeker/standard-RAxML/raxmlHPC"
    RATE4SITE_COMMAND_PREFIX = "/groups/pupko/noaeker/rate4site/rate4site"
elif LOCAL_RUN:
    IQTREE_EXE = "/Users/noa/Programs/iqtree-2.1.3-MacOSX/bin/iqtree2"
    RAXML_NG_EXE = "/Users/noa/Programs/Raxml/raxml-ng  "
    RAXML_HPC_EXE = "/Users/noa/Programs/standard-RAxML/raxmlHPC-PTHREADS "
    MAD_COMMAND_PREFIX = "/Users/noa/Programs/mad.osx"
    RATE4SITE_COMMAND_PREFIX = "/Users/noa/Programs/rate4site"
    RESULTS_FOLDER = "/Users/noa/Workspace/raxml_deep_learning_results"
    MSAs_CSV_PATH = "/Users/noa/Workspace/data/sampled_datasets.csv"
    GENERAL_MSA_DIR = "/Users/noa/Workspace/data/single-gene_alignments"
    MAIN_CODE_PATH = "/Users/noa/Workspace/RAxML_params_tuning/tune_params.py"
