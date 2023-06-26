"""Python script to train the model"""
import joblib
import mne
import numpy as np
from prefect import flow, task
from scipy.signal import resample
from sklearn.decomposition import NMF

from config import Location
from utils import (find_valid_peaks, get_raw_data, get_thresholds, load_config,
                   remove_border_spikes)


@task
def get_preprocessed_data(data_location: str):
    """Get preprocessed data from a specified location

    Parameters
    ----------
    data_location : str
        Location to get the data
    """
    return joblib.load(data_location)


@task
def load_model(load_path: str):
    """Load model (i.e., weights) from a specified location

    Parameters
    ----------
    load_path : str
    """
    return joblib.load(load_path)


@task
def detect_spikes(preprocessed_data, W, max_spike_freq, thresholds, H_freq=50):

    # Provide fitted weights W to scikit-learn NMF to get activation score H
    nmf = NMF(init='custom', n_components=W.shape[1])
    H, _, _ = nmf._fit_transform(X=preprocessed_data['ll_data'].T, H=W.T, update_H=False)
    H = H.T

    # Iterate through each H (each for one basis function)
    for base_idx in range(H.shape[0]):

        # Find peaks that are at least max_spike_freq seconds apart
        peak_indices, peak_heights = find_valid_peaks(
            H[base_idx], H_freq, max_spike_freq=max_spike_freq, height=thresholds[base_idx]  # TODO parameterise
        )


@flow
def detect(
    location: Location = Location(),
):
    """Flow to train the model

    Parameters
    ----------
    location : Location, optional
        Locations of inputs and outputs, by default Location()
    """
    thresholds = load_config(location.thresholds)['thresholds']
    preprocess_config = load_config(location.detection_config)['preprocess']

    preprocessed_data = get_preprocessed_data(location.data_preprocess)
    W = load_model(location.model)
    spikes = detect_spikes(preprocessed_data, W, thresholds)
    spikes, preprocess_config


if __name__ == "__main__":
    detect()
