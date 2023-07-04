"""Python script to preprocess the data"""

import math
import os
import sys

import joblib
import mne
import numpy as np
from prefect import flow, task
from scipy.signal.windows import hann

from utils import copy_file, get_raw_data, load_config


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


def hanning_window(raw_data):
    """Apply Hanning window to raw EEG data.

    Apply a one-second taper to either side of the raw EEG data.

    Parameters
    ----------
    raw_data : mne.io.RawEDF
        The raw EEG data to apply the Hanning window to. The sampling frequency
        information should be accessed through `raw_data.info["sfreq"]`.

    Returns
    -------
    mne.io.RawArray
        The EEG data with the Hanning window applied.

    Notes
    -----
    The Hanning window is a weighted windowing function used in signal processing.
    It tapers the edges of the input data to minimize spectral leakage when performing
    Fourier transforms or analyzing frequency content.

    The function applies a 2-second taper, with a frequency value calculated from the
    sampling frequency (`raw_data.info["sfreq"]`) on either side of the EEG data.
    The left and right side windows are multiplied by the respective halves of the Hanning
    window, and the result is returned as an `mne.io.RawEDF` object.
    """

    # Extract the EEG data
    data = raw_data.get_data()

    # Calculate the sampling frequency
    sfreq = int(raw_data.info["sfreq"])

    # Calculate the size of the taper window
    w = 2 * sfreq

    # Generate the Hanning window
    haw = 1 - hann(w)

    # Extract the left and right side windows from the Hanning window
    lhaw = haw[sfreq:]
    rhaw = haw[:sfreq]

    # Apply the Hanning window to the left and right sides of the EEG data
    data[:, :sfreq] = np.multiply(data[:, :sfreq], lhaw)
    data[:, -sfreq:] = np.multiply(data[:, -sfreq:], rhaw)

    # Return the modified EEG data as an mne.io.RawArray object
    return mne.io.RawArray(data, raw_data.info)


@flow
def preprocess(
    raw_data_path: str,
    detection_config: str = "detection_config.yaml",
    batch_len: int = 900
):
    """
    Preprocess the raw data, then save this preprocessed data along with the configuration file.
    This is done batch wise, with batches of 1hr by default.

    Parameters:
    -----------
    raw_data_path : str
        The path to the raw edf file.
    detection_config : str, optional
        The path to the configuration file (default is "detection_config.yaml").
    batch_len : int, optional
        The length of a batch in seconds. Default is 900 (15 minutes).

    Example:
    ---------
    preprocess('path/to/raw_data.edf', 'detection_config.yaml')

    Returns:
    ________
    tmp_data_folder : str
        Path to where config file and was copied to, and where preprocessed data is stored
    """

    # Load the relevant configuration parameters
    preprocess_config = load_config(detection_config)['preprocess']

    # Load the raw data
    data = get_raw_data(raw_data_path, preload=False)

    # Get length of recording in seconds
    data_len = data.times[-1] - data.times[0]

    # Stores all preprocessed data
    preprocessed_data = []

    # Calculate number of required batches
    n_batches = math.ceil(data_len / batch_len)

    # Loop through batches
    for batch_idx in range(n_batches):

        # Create copy of data
        batch_data = data.copy()

        # Get start and end times for current batch
        tmin = batch_len * batch_idx
        tmax = data_len if tmin + batch_len > data_len else tmin + batch_len

        # Crop to appropriate time
        batch_data.crop(tmin=tmin, tmax=tmax)

        # Preprocess the data
        preprocessed_batch = preprocess_data(batch_data, preprocess_config)

        # Apply hanning window with 1 second taper to remove discontinuity
        batch_data = hanning_window(batch_data)

        # Add to list of all preprocessed batches
        preprocessed_data.append(preprocessed_batch)

    # Convert from list to np array
    preprocessed_data = np.concatenate(preprocessed_data, axis=1)

    # Obtain the folder path for the raw data file, and create subdirectory called tmp
    tmp_data_folder = os.path.join('data', 'processed')
    if not os.path.exists(tmp_data_folder):
        os.makedirs(tmp_data_folder)

    # Save the preprocessed data in the same folder as the raw data, as "data_processed.pkl"
    save_preprocessed_data(preprocessed_data, os.path.join(tmp_data_folder, "data_processed.pkl"))

    # Get path for where config file should be copied to: same folder as the preprocessed data
    config_path = os.path.join(tmp_data_folder, os.path.basename(detection_config))

    # Copy the config file
    copy_file(detection_config, config_path)

    return tmp_data_folder


if __name__ == "__main__":
    preprocess(**dict(arg.split('=') for arg in sys.argv[1:]))
