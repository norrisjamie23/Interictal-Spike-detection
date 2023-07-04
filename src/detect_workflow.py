"""Python script to preprocess an edf file and run detection on it given directory with model weights and some thresholds"""

import sys
from pathlib import Path

from detect_spikes import detect
from preprocess import preprocess


def detect_flow(raw_data_path: str, model_dir: str) -> None:
    """
    Perform seizure detection on preprocessed data using a specified model.

    Parameters
    ----------
    raw_data_path : str
        The path to the raw data file.
    model_dir : str
        The directory containing the detection model.

    Returns
    -------
    None
    """

    # Preprocess data, saving to returned directory (tmp_data_folder)
    tmp_data_folder = preprocess(raw_data_path)

    # The detections will be saved here
    # A subdirectory with the model name is created (if it doesn't exist) with the model name
    # The detection .csv is stored in this file with the name of the edf file (but .csv instead of .edf)
    save_path = f'{Path(raw_data_path).parent / Path(model_dir).stem / Path(raw_data_path).stem}.csv'

    # Perform detection and save to aforementioned location
    detect(tmp_data_folder, model_dir, save_path)


if __name__ == "__main__":
    detect_flow(**dict(arg.split('=') for arg in sys.argv[1:]))
