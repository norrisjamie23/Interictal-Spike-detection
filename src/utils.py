"""Collection of Python functions used elsewhere"""

import csv
import os
import shutil
from pathlib import Path

import mne
import numpy as np
import yaml
from prefect import task
from scipy.signal import find_peaks, savgol_filter


@task
def get_raw_data(data_location: str, preload: bool = True):
    """Read raw data

    Parameters
    ----------
    data_location : str
        The location of the raw data
    """
    return mne.io.read_raw_edf(data_location, preload=preload)


def get_minimum_thresh(H: np.ndarray, k=1):
    """Compute the threshold to use for scipy.signal.find_peaks

    Parameters
    ----------
    H : np.ndarray
        The activation function for a single basis function
    k : int, optional
        Scaling factor for determining the threshold (default is 1)

    Returns
    -------
    threshold : float
        The computed threshold value for detection
    """

    # Determine the number of bins
    nbin = min(round(0.1 * len(H)), 1000)

    # Compute the probability density function (PDF) using histogram
    n, x = np.histogram(H, nbin)

    # Convert bin edges to bin centers
    x = (x[1:] + x[:-1]) / 2

    # Exclude the first five bins (can be high if many bad times)
    n = n[5:]
    x = x[5:]

    # Smooth the PDF using a third-degree polynomial fit
    n_s = savgol_filter(n, 11, 3)

    # Exclude the last 11 points as they can be skewed by smoothing
    n = n[:-11]
    n_s = n_s[:-11]
    x = x[:-11]

    # Find the position of the mode/peak of the PDF
    modn = np.argmax(n_s)

    # Compute the first derivative (max difference between bins)
    d1 = np.diff(n_s)
    d1 = np.insert(d1, 0, d1[0], axis=0)
    d1_s = savgol_filter(d1, 11, 3)

    # Find the position of the minimum on the right of the mode
    mr = np.argmin(d1_s[modn:])
    mr = mr + modn

    # Compute the second derivative (max relative difference between bins)
    d2 = np.diff(d1_s)
    d2 = np.insert(d2, 0, d2[0], axis=0)

    # Smooth the second derivative (not sure if needed, decide whether to remove this line and amend line below)
    d2_s = savgol_filter(d2, 11, 3)

    # Find the position of the second maxima of the second derivative
    ix = np.argmax(d2_s[mr:])
    ix = ix + mr

    # Compute the threshold as k times the distance between mr and ix
    threshold = x[ix + k * (ix - mr)]

    return threshold


def find_valid_peaks(activation: np.ndarray, H_freq: int, max_spike_freq=0.5, height='auto'):
    """Find valid peaks in the activation function.

    Parameters
    ----------
    activation : np.ndarray
        The activation function for a single basis function.
    activation_freq : int
        The frequency (or sample rate) for the activation time-series.
    max_spike_freq : number
        The maximum frequency at which spikes are expected to occur (in datapoints)
    height : number or 'auto'
        Required height of peaks in H. If 'auto', found using get_minimum_thresh. Use None if not a valid cluster.

    Returns
    -------
    indices : array-like
        The indices of the identified peaks.
    peak_heights : array-like
        The corresponding heights of the identified peaks.
    """

    # If threshold is None, return empty list
    if not height:
        return [], []
    elif height == 'auto':
        # If auto, determine minimum threshold to use
        height = get_minimum_thresh(activation)

    # Minimum number of datapoints between detections (default: 0.3s)
    distance = max_spike_freq * H_freq

    #  With SciPy, identify peaks that exceed minimum threshold
    peaks = find_peaks(activation, height=height, distance=distance)

    return peaks[0], peaks[1]["peak_heights"]


def remove_border_spikes(
    peak_indices: np.ndarray,
    peak_heights: np.ndarray,
    H_freq: int,
    context: int,
    activation_len: int,
):
    """Remove spikes that occur within the first and last context seconds of the activation function.

    Parameters
    ----------
    peak_indices : np.ndarray
        The indices of the identified peaks.
    peak_heights : np.ndarray
        The corresponding heights of the identified peaks.
    H_freq : int
        The frequency (or sample rate) at which the activation function H is recorded.
    context : int
        The number of seconds of context to exclude from the beginning and end of the activation function.
    activation_len : int
        The length of the activation function (in datapoints).

    Returns
    -------
    peak_indices : array-like
        The indices of the identified peaks after removing spikes within the border.
    peak_heights : array-like
        The corresponding heights of the identified peaks after removing spikes within the border.
    """

    # Get peaks outside of first 5s
    peak_heights = peak_heights[peak_indices > H_freq * context]
    peak_indices = peak_indices[peak_indices > H_freq * context]

    # Get peaks outside of last 5s
    peak_heights = peak_heights[
        peak_indices < activation_len - H_freq * context
    ]
    peak_indices = peak_indices[
        peak_indices < activation_len - H_freq * context
    ]

    return peak_indices, peak_heights


def get_thresholds(
    peak_heights: np.ndarray,
    num_thresholds: int = 10,
    log_transform: bool = True,
):
    """Compute a range of potential thresholds for identifying interictal spikes.

    Parameters
    ----------
    peak_heights : np.ndarray
        The heights of the potential peaks corresponding to interictal spikes.
    num_thresholds : int, optional
        The number of thresholds to generate (default is 10).
    log_transform : bool, optional
        Flag indicating whether to apply a logarithmic transformation to the peak heights (default is True).

    Returns
    -------
    thresholds : np.ndarray
        An array of potential thresholds for identifying interictal spikes.

    Notes
    -----
    This function generates a range of potential thresholds based on the heights of potential peaks.
    It offers two options for threshold generation:
    1. Linear spacing: If log_transform is set to False, thresholds are linearly spaced across the range of heights.
    2. Logarithmic spacing: If log_transform is set to True, thresholds are logarithmically spaced.
    """

    if log_transform:
        # Apply logarithmic transformation to the peak heights
        peak_heights = np.log(peak_heights)

    # Compute the step size between thresholds
    step_size = (np.max(peak_heights) - np.min(peak_heights)) / (
        num_thresholds + 1
    )

    # Generate the thresholds
    thresholds = np.arange(
        np.min(peak_heights) + step_size, np.max(peak_heights), step_size
    )

    if log_transform:
        # Convert the thresholds back to the original scale using exponential function
        thresholds = np.exp(thresholds)

    # Return a subset of thresholds based on the specified num_thresholds parameter
    return thresholds[:num_thresholds]


def load_config(filename):
    """
    Loads a YAML configuration file and returns its contents as a Python object.

    Parameters:
    -----------
    filename : str
        The path to the YAML configuration file.

    Returns:
    --------
    dict or list
        The contents of the YAML file as a Python object.

    Raises:
    -------
    FileNotFoundError
        If the specified file does not exist.
    yaml.YAMLError
        If there is an error in parsing the YAML file.

    Example:
    ---------
    config_data = load_config('config.yaml')
    """

    with open(filename, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    return config


def save_list_as_csv(list_to_save: list, filename: str):
    """
    Saves a list as a CSV file.

    Parameters:
    -----------
    list_to_save : list
        The list that should be saved.
    filename : str
        The path to the CSV file. The directory is created if it doesn't exist.
    """

    # Create parent folder if it doesn't exist
    create_directory(filename, file=True)

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(list_to_save)


def create_directory(directory: str, file: bool = False):

    """
    Create a directory, or optionally create a parent directory for a file that is to be created.

    Parameters:
    -----------
    filename : str
        The file that is to be created - parent of this is created if it doesn't exist.
    file : bool
        Setting this to true instead creates the parent directory. Set to true if directory is a file.
    """

    # If directory is a file, and we want to create the parent folder
    if file:
        os.makedirs(Path(directory).parent, exist_ok=True)
    # Else if directory is a folder, and we want to create it
    else:
        os.makedirs(Path(directory), exist_ok=True)


def copy_file(src: str, dst: str):
    """
    Copies a file from the source path to the destination path.

    Parameters:
    -----------
    src : str
        The source path of the file to be copied.
    dst : str
        The destination path where the file will be copied.
    """

    # Copy file from src to dst
    shutil.copyfile(src, dst)
