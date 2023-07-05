"""Python script to detect spikes given model weights and some thresholds"""

from pathlib import Path

import joblib
import numpy as np
from prefect import flow, task
from sklearn.decomposition import NMF

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
    preprocessed_data_dir: str,
    model_dir: str,
    save_path: str
) -> None:

    """
    Perform the flow to detect spikes in preprocessed data using a trained model.

    Parameters
    ----------
    preprocessed_data_dir : str
        The directory path where the preprocessed data is stored.
    model_dir : str
        The directory path where the trained model and configuration files are stored.
    save_path : str
        The file path where the spike detections will be saved.
    """

    # Get path to thresholds from model_dir, then load thresholds
    thresholds_path = Path(model_dir) / Path("thresholds.yaml")
    thresholds = load_config(thresholds_path)['thresholds']

    # Get path to config file from model_dir, then load config
    config_path = Path(model_dir) / Path("detection_config.yaml")
    config = load_config(config_path)['preprocess']

    # Get path to preprocessed data, then load
    preprocessed_data_path = Path(preprocessed_data_dir) / Path("data_processed.pkl")
    preprocessed_data = get_preprocessed_data(preprocessed_data_path)

    # Get model path and load model
    model_path = Path(model_dir) / Path("nmf_weights.pkl")
    W = load_model(model_path)

    # Detect spikes from preprocessed data using model, config, and thresholds
    detections = detect_spikes(preprocessed_data, W, config['max_spike_freq'], thresholds)

    # Save detections as save_path
    save_list_as_csv(detections, save_path)
