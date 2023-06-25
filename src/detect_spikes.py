"""Python script to train the model"""
import joblib
import mne
import numpy as np
from prefect import flow, task
from scipy.signal import resample
from sklearn.decomposition import NMF

from config import Location, ModelParams, PreprocessParams
from utils import (find_valid_peaks, get_raw_data, get_thresholds,
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
def detect_spikes(preprocessed_data, W, preprocess_params, thresholds):

    # Provide fitted weights W to scikit-learn NMF to get activation score H
    nmf = NMF(init='custom', n_components=W.shape[1])
    H, _, _ = nmf._fit_transform(X=preprocessed_data['ll_data'].T, H=W.T, update_H=False)
    H = H.T

    # Iterate through each H (each for one basis function)
    for base_idx in range(H.shape[0]):

        # Find peaks that are at least max_spike_freq seconds apart
        peak_indices, peak_heights = find_valid_peaks(
            H[base_idx], preprocess_params.H_freq, max_spike_freq=0.3, height=thresholds[base_idx]  # TODO parameterise
        )


@flow
def detect(
    location: Location = Location(),
    nmf_params: ModelParams = ModelParams(),
    preprocess_params: PreprocessParams = PreprocessParams(),
):
    """Flow to train the model

    Parameters
    ----------
    location : Location, optional
        Locations of inputs and outputs, by default Location()
    svc_params : ModelParams, optional
        Configurations for training the model, by default ModelParams()
    """
    preprocessed_data = get_preprocessed_data(location.data_preprocess)
    W = load_model(location.model)
    spikes = detect_spikes(preprocessed_data, W, preprocess_params)
    spikes


if __name__ == "__main__":
    detect()
