"""Python script to preprocess an edf file and train an NMF model on it"""

import sys

import numpy as np
from prefect import flow, task
from scipy.signal import le, resamp

from utils import (find_valid_peaks, get_raw_data, get_thresholds, load_config,
                   remove_border_spikes)


def train_flow(
    raw_data_path,
):
    pass
    # python src/preprocess.py $(raw_data_path_opt)
    # python src/train_model.py config_path=$$(config_path)


if __name__ == "__main__":
    train_flow(**dict(arg.split('=') for arg in sys.argv[1:]))
