"""Python script to process the data"""

import joblib
import mne
import numpy as np
from prefect import flow, task

from config import Location, PreprocessConfig
from utils import get_raw_data


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
def process_data(data: mne.io.edf.edf.RawEDF, config):
    """Apply pre-processing, including line-length transformation

    Parameters
    ----------
    data : mne.io.edf.edf.RawEDF
        Data to process.
    config : dict
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
    data.filter(0.1, config.highpass_freq)

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

    # Notch filter from 50 Hz, up in 50 Hz increments
    notch_filter_freqs = np.arange(50, config.highpass_freq, 50)
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
    data.resample(sfreq=config.H_freq)

    # Get data as a NumPy array
    ll_data = data.get_data()

    # Resampling can result in negative values, set these to 0
    ll_data[ll_data < 0] = 0

    return ll_data


@task
def package_data(ll_data, highpass_freq, H_freq):
    """Package line-length transformed data for saving.

    Parameters
    ----------
    ll_data : np.ndarray
        Line-length transformed data.
    highpass_freq : int
        High-pass filter frequency.
    H_freq : int
        Resampling frequency.

    Returns
    -------
    dict
        Dictionary containing the packaged data and parameters.

    """
    return {
        "ll_data": ll_data,
        "highpass_freq": highpass_freq,
        "H_freq": H_freq,
    }


@task
def save_processed_data(data: dict, save_location: str):
    """Save processed data

    Parameters
    ----------
    data : dict
        Data to process
    save_location : str
        Where to save the data
    """
    joblib.dump(data, save_location)


@flow
def process(
    location: Location = Location(),
    config: PreprocessConfig = PreprocessConfig(),
):
    """Flow to process the ata

    Parameters
    ----------
    location : Location, optional
        Locations of inputs and outputs, by default Location()
    config : PreprocessConfig, optional
        Configurations for processing data, by default PreprocessConfig()
    """
    data = get_raw_data(location.data_raw)
    processed_data = process_data(data, config)
    dict_data = package_data(processed_data, config)
    save_processed_data(dict_data, location.data_process)


if __name__ == "__main__":
    process(config=PreprocessConfig())
