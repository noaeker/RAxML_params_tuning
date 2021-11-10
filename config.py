import logging
import random

LOCAL_RUN = True
MODULE_LOAD_STR = "module load gcc/gcc-8.2.0; module load R/3.6.1; module load python/python-anaconda3.6.5-orenavr2; module load intel/parallel_studio_xe_2020.4.omnipath;"
SEED =1


BASELINE = "X"
CURR_RUN_PREFIX = "first_tuning"
CURR_JOBS_PREFIX = "first_tuning"
N_CPUS_PER_JOB= 1
N_CPUS_PER_TRAINING= 1
N_CPUS_RAXML = 1
N_MSAS = 2
N_JOBS = 1
N_SEQ = 10
MIN_N_SEQ = 10
N_PARSIMONY_RAXML_SEARCH= 1
N_RANDOM_RAXML_SEARCH = 1
N_LOCI = 100
MIN_N_LOCI = 100
LOGGING_LEVEL = logging.INFO
OUTPUT_CSV_NAME = "tune_raxml"
WAITING_TIME_UPDATE = 60
TRAINING_SIZE = 1000
EPSILON = 0.1


def sample_exp(size,start_seed):
    res = []
    seed = start_seed
    for i in range(size):
        random.seed(seed)
        res.append(random.expovariate(lambd=10))
        seed = seed + 1
    return res
BRLEN_GENERATORS = {'exponential':sample_exp,'optimized': None}



if not LOCAL_RUN:
    # PATH CONFIGURATION
    RAXML_NG_EXE = "/groups/pupko/noaeker/raxml-ng-float-mpi/raxml-ng --extra thread-pin "
    MAD_COMMAND_PREFIX = "/groups/pupko/noaeker/mad"
    RESULTS_FOLDER = "/groups/pupko/noaeker/lasso_positions_sampling_results"
    MSAs_FOLDER = "/groups/pupko/noaeker/data/ABC_DR"
    MSAs_CSV_PATH = "/groups/pupko/noaeker/data/sampled_datasets.csv"
    ALTERNATIVER_FILES_FOLDER = "/groups/pupko/noaeker/example"
    MAIN_CODE_PATH = "/groups/pupko/noaeker/lasso_positions_sampling/parallel_code/MSA_positions_sampling.py"
    R_CODE_PATH = "/groups/pupko/noaeker/lasso_positions_sampling/R_code/lasso_glmnet.R"
    RAXML_HPC_EXE = "/groups/pupko/noaeker/standard-RAxML/raxmlHPC"
    RATE4SITE_COMMAND_PREFIX = "/groups/pupko/noaeker/rate4site/rate4site"
elif LOCAL_RUN:
    IQTREE_PATH = "/Users/noa/Programs/iqtree-1.6.12-MacOSX/iqtree"
    RAXML_NG_EXE = "/Users/noa/Programs/Raxml/raxml-ng  "
    RAXML_HPC_EXE = "/Users/noa/Programs/standard-RAxML/raxmlHPC-PTHREADS "
    MAD_COMMAND_PREFIX = "/Users/noa/Programs/mad.osx"
    RATE4SITE_COMMAND_PREFIX = "/Users/noa/Programs/rate4site"
    RESULTS_FOLDER= "/Users/noa/Workspace/raxml_deep_learning_results"
    MSAs_FOLDER = "/Users/noa/Workspace/data/ABC_DR"
    MSAs_CSV_PATH = "/Users/noa/Workspace/data/sampled_datasets.csv"
    ALTERNATIVER_FILES_FOLDER= "/Users/noa/Workspace/data/supermatrices_edited"
    MAIN_CODE_PATH = "/Users/noa/Workspace/RAxML_params_tuning/tune_params.py"
