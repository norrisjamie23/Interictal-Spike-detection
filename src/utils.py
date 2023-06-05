"""Python script to process the data"""

import mne
import numpy as np
from prefect import flow, task
from scipy.signal import find_peaks, savgol_filter


@task
def get_raw_data(data_location: str):
    """Read raw data

    Parameters
    ----------
    data_location : str
        The location of the raw data
    """
    return mne.io.read_raw_edf(data_location, preload=True)


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


def find_valid_peaks(
    activation: np.ndarray, activation_freq: int, max_spike_freq
):
    """Find valid peaks in the activation function.

    Parameters
    ----------
    activation : np.ndarray
        The activation function for a single basis function.
    activation_freq : int
        The frequency (or sample rate) for the activation time-series.
    max_spike_freq : number
        The maximum frequency at which spikes are expected to occur (in datapoints)

    Returns
    -------
    indices : array-like
        The indices of the identified peaks.
    peak_heights : array-like
        The corresponding heights of the identified peaks.
    """

    # Get minimum threshold to use
    height = get_minimum_thresh(activation)

    # Minimum number of datapoints between detections (default: 0.3s)
    distance = max_spike_freq * activation_freq

    #  With SciPy, identify peaks that exceed minimum threshold
    peaks = find_peaks(activation, height=height, distance=distance)

    return peaks[0], peaks[1]["peak_heights"]
