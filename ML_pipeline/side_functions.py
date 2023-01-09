from side_code.config import *
import argparse
import numpy as np


def get_ML_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline_folder', action='store', type=str, default=f"{READY_RAW_DATA}/Pandit/ML")
    parser.add_argument('--val_pct', action='store', type=int,
                        default=0)
    parser.add_argument('--test_pct', action='store', type=int,
                        default=0.3)
    parser.add_argument('--n_jobs', action='store', type=int,
                        default=4)
    parser.add_argument('--test_different_training_sizes',action='store_true', default = False)
    parser.add_argument('--different_training_sizes', default=np.linspace(0.1,1,5))
    parser.add_argument('--large_grid',action='store_true', default=False)
    parser.add_argument('--do_RFE', action='store_true', default=False)
    parser.add_argument('--filter_pandit', action='store_true', default=False)
    parser.add_argument('--n_CV_folds',type=int, default = 3)
    parser.add_argument('--name', type=str, default="new_run")
    parser.add_argument('--n_sample_points', type=int, default=100)
    parser.add_argument('--filter_unreliables', type=int, default=100)
    return parser