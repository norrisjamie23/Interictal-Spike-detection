"""Python script to preprocess the data"""

import os
import sys

import joblib
import mne
import numpy as np
from prefect import flow, task

from utils import copy_file, file_to_folder, get_raw_data, load_config


def line_length(a, w=20):
    """Compute the line-length for each channel of EEG.

    Parameters
    ----------
    a : NumPy array
        The data to be transformed
    w : int
        The line-length window, i.e., the number of datapoints to use

    Returns
    -------
    NumPy array
        The line-length values for each channel of the input data
    """

    # Get absolute difference between datapoints
    a = np.abs(np.diff(a))

    # Pad to appropriate length
    a = np.pad(a, ((0, 0), (w // 2 + 1, w // 2 + w % 2)), "constant")

    # Compute cumulative sum along the axis of channels
    ret = np.cumsum(a, axis=1, dtype=float)

    # Calculate line-length values by subtracting cumulative sums at the beginning and end of the window
    return ret[:, w:] - ret[:, :-w]


@task
def preprocess_data(data: mne.io.edf.edf.RawEDF, preprocess_config: dict):
    """Apply pre-processing, including line-length transformation

    Parameters
    ----------
    data : mne.io.edf.edf.RawEDF
        Data to preprocess.
    preprocess_config : dict
        Configuration parameters.

        Returns
    -------
    np.ndarray
        Line-length transformed data.

    """

    # Get data as a NumPy array
    raw = data.get_data()

    # Remove occassional DC component by zero-centering individual channels
    raw -= np.expand_dims(np.nanmedian(raw, axis=1), axis=-1)

    # Create a new RawEDF object with the zero-centred data
    data = mne.io.RawArray(raw, data.info)

    # Bandpass filter between 0.1 and highpass_freq (Default: 50 Hz)
    data.filter(0.1, preprocess_config['highpass_freq'])

    # Get data as a NumPy array
    raw = data.get_data()

    # Re-reference to the global median voltage
    raw -= np.median(raw)

    # This is set as 50 µV in Maxime's paper
    scaled_voltage = 50

    # Normalise each channel by scaling its median absolute voltage to 50 µV
    raw *= np.expand_dims(
        (scaled_voltage / np.median(np.absolute(raw), axis=1)), axis=-1
    )

    # Create a new RawEDF object with the transformed data
    data = mne.io.RawArray(raw, data.info)

    # Notch filter from powerline_freq (default: 50 Hz), up in 50 Hz increments
    notch_filter_freqs = np.arange(preprocess_config['powerline_freq'], preprocess_config['highpass_freq'], preprocess_config['powerline_freq'])
    if len(notch_filter_freqs) > 0:
        data.notch_filter(notch_filter_freqs)

    # Resample to 500 Hz
    data.resample(sfreq=500)

    # 40ms horizon
    window = 40

    # Get line-length window (40ms) in number of datapoints
    window_pts = int(np.round(data.info["sfreq"] * window / 1000))

    # Apply line-length transformation to data
    ll_data = line_length(data.get_data(), w=window_pts)

    # Create a new RawEDF object with the line-length transformed data
    data = mne.io.RawArray(ll_data, data.info)

    # Low-pass filter at 20 Hz
    data.filter(None, 20)

    # Resample (default: 50 Hz)
    data.resample(sfreq=50)

    # Get data as a NumPy array
    ll_data = data.get_data()

    # Resampling can result in negative values, set these to 0
    ll_data[ll_data < 0] = 0

    return ll_data


@task
def save_preprocessed_data(data: dict, save_location: str):
    """Save preprocessed data

    Parameters
    ----------
    data : dict
        Data to preprocess
    save_location : str
        Where to save the data
    """
    joblib.dump(data, save_location)


@flow
def preprocess(
    raw_data_path: str,
    detection_config: str = "detection_config.yaml"
):
    """
    Preprocess the raw data, then save this preprocessed data along with the configuration file.

    Parameters:
    -----------
    raw_data_path : str
        The path to the raw edf file.
    detection_config : str, optional
        The path to the configuration file (default is "detection_config.yaml").

    Example:
    ---------
    preprocess('path/to/raw_data.edf', 'detection_config.yaml')
    """

    # Load the relevant configuration parameters
    preprocess_config = load_config(detection_config)['preprocess']

    # Load the raw data
    data = get_raw_data(raw_data_path)

    # Preprocess the data
    preprocessed_data = preprocess_data(data, preprocess_config)

    # Obtain the folder path for the raw data file
    data_folder = file_to_folder(raw_data_path)

    # Save the preprocessed data in the same folder as the raw data, as "data_processed.pkl"
    save_preprocessed_data(preprocessed_data, os.path.join(data_folder, "data_processed.pkl"))

    # Copy the configuration file to the same folder as the preprocessed data
    copy_file(detection_config, os.path.join(data_folder, os.path.basename(detection_config)))


if __name__ == "__main__":
    preprocess(**dict(arg.split('=') for arg in sys.argv[1:]))
