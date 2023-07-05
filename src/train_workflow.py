"""Python script to preprocess an edf file and train an NMF model on it"""

import sys

from preprocess import preprocess
from train_model import train


def train_flow(raw_data_path: str) -> None:
    """
    Preprocess an edf file and train an NMF model on it.

    Parameters
    ----------
    raw_data_path : str
        The path to the raw data file.
    """

    # Preprocess data, saving to returned directory (tmp_data_folder)
    tmp_data_folder = preprocess(raw_data_path)

    # Train a model on this preprocessed data, saving all relevant files to a subdirectory of raw_data_path
    train(tmp_data_folder, raw_data_path)


if __name__ == "__main__":
    train_flow(**dict(arg.split('=') for arg in sys.argv[1:]))
