"""Python script to train the model"""
import joblib
import numpy as np
from prefect import flow, task
from sklearn.decomposition import NMF

from config import Location
from utils import find_valid_peaks, load_config, save_list_as_csv


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
def detect_spikes(preprocessed_data: np.ndarray, W: np.ndarray, max_spike_freq: float, thresholds: list, H_freq=50):
    """
    Detects interictal spikes from NMF weights and line-length transformed data.

    Parameters:
    -----------
    preprocessed_data : np.ndarray
        Preprocessed data used for spike detection.
    W : np.ndarray
        NMF weights representing basis functions.
    max_spike_freq : float
        Maximum frequency (in seconds) allowed between consecutive spikes.
    thresholds : list
        List of threshold values corresponding to each basis function.
    H_freq : int, optional
        Frequency (in Hz) of the H activation score. Default is 50.

    Returns:
    --------
    list
        A list containing two lists: the first list represents the time of spike detections,
        and the second list represents the cluster indices.

    Example:
    ---------
    detections = detect_spikes(preprocessed_data, W, max_spike_freq=0.5, thresholds=[None, 0.5, 0.7, None, 0.4])
    """

    # Provide fitted weights W to scikit-learn NMF to get activation score H
    nmf = NMF(init='custom', n_components=W.shape[1])
    H, _, _ = nmf._fit_transform(X=preprocessed_data.T, H=W.T, update_H=False)
    H = H.T

    # List to store all detected spikes across all clusters
    # First list is time of spike, second list is cluster
    detections = [[], []]

    # Iterate through each H (each for one basis function)
    for base_idx in range(H.shape[0]):

        # Find peaks that are at least max_spike_freq seconds apart
        peak_indices, _ = find_valid_peaks(
            H[base_idx], H_freq, max_spike_freq=max_spike_freq, height=thresholds[base_idx]
        )

        # If list isn't empty (if, if any spikes were detected)
        if len(peak_indices) > 0:
            # Convert to seconds and append to detections[0]
            detections[0].extend(list(peak_indices / H_freq))

            # Add cluster index to detections[1]
            detections[1].extend([base_idx] * len(peak_indices))

    return detections


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

    detections = detect_spikes(preprocessed_data, W, preprocess_config['max_spike_freq'], thresholds)
    save_list_as_csv(detections, location.detections)


if __name__ == "__main__":
    detect()
