from side_code.config import *
import argparse
import numpy as np


def get_ML_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline_folder', action='store', type=str, default=f"{READY_RAW_DATA}/Pandit/ML")
    parser.add_argument('--n_sample_points', action='store', type=int,
                        default=100)
    parser.add_argument('--tree_choosing_method', action='store', type=str,
                        default='ML')
    parser.add_argument('--val_pct', action='store', type=int,
                        default=0.25)
    parser.add_argument('--test_pct', action='store', type=int,
                        default=0.25)
    parser.add_argument('--max_starting_trees', action='store', type=int,
                        default=40)
    parser.add_argument('--clusters_max_dist_options', action='store', type=float,
                        default=[0]) #np.linspace(0,1,10)
    parser.add_argument('--n_jobs', action='store', type=int,
                        default=4)
    parser.add_argument('--lightgbm', action='store_true', default=True)
    parser.add_argument('--fast_run', action='store_true', default=False)
    return parser