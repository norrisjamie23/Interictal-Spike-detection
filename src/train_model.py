"""Python script to train a model on a preprocessed edf file"""

from datetime import datetime
from pathlib import Path

import joblib
import mne
import nimfa
import numpy as np
from prefect import flow, task
from scipy.signal import resample

from utils import (copy_file, create_directory, find_valid_peaks, get_raw_data,
                   get_thresholds, load_config, remove_border_spikes)


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
def train_model(rank: int, ll_data: np.ndarray):
    """Train the model using NMF (Nonnegative Matrix Factorization)

    Parameters
    ----------
    rank : int
        Rank for NMF model
    ll_data : np.ndarray
        Line-length transformed data to use for training

    Returns
    -------
    W : np.ndarray
        Matrix of basis vectors (weights)
    H : np.ndarray
        Matrix of activation scores (coefficients)
    """

    print(f"Training NMF model with rank {rank}")

    # Perform NMF with multiple runs and multiplicative updates
    nmf = nimfa.Nmf(
        ll_data, max_iter=5, rank=rank, n_run=30
    )
    nmf_fit = nmf()

    # Print RSS of best model
    print("Initial model RSS: %5.4f" % nmf_fit.fit.rss())

    # Get W and H to initialise next NMF (Alternating Nonnegative Least Squares Matrix Factorization)
    W = nmf_fit.basis()
    H = nmf_fit.coef()

    # Fit a single model, initialising from the best prior model,
    # up to either 1000 iterations or min_residual of 1e-4
    lsnmf = nimfa.Lsnmf(
        ll_data,
        seed="fixed",
        W=W,
        H=H,
        rank=rank,
        max_iter=1000,
        min_residuals=1e-4,
    )
    lsnmf_fit = lsnmf()

    # Print RSS of final model
    print("Final model RSS: %5.4f" % lsnmf_fit.fit.rss())

    # Get weights (W) and activation scores (H)
    W = np.array(lsnmf_fit.basis())
    H = np.array(lsnmf_fit.coef())

    return W, H


@task
def save_spikes_for_labelling(
    original_data: mne.io.edf.edf.RawEDF,
    W: np.ndarray,
    H: np.ndarray,
    max_spike_freq: float,
    save_location: str,
    num_chans: int = 20,
    spikes_per_cluster: int = 10,
    context: int = 5,
    H_freq=50,
):
    """
    Save potential spikes for manual labelling.

    Parameters
    ----------
    original_data : RawEDF
        The original EEG data.
    W : ndarray
        The weight matrix.
    H : ndarray
        The spike activation matrix.
    max_spike_freq : float
        The maximum frequency of spikes in Hz (default is 0.3).
    save_location : str
        The file path to save the processed data.
    num_chans : int, optional
        The number of channels to consider (default is 20).
    spikes_per_cluster : int, optional
        The number of spikes to select per cluster (default is 10).
    context : int, optional
        The context duration in seconds around each spike (default is 5).
    H_freq : int, optional
        Frequency of activation matrix H.
    """

    # Lists to use for annotations for potential spikes in new file
    onsets = []
    descriptions = []
    all_potential_spikes = []

    # Iterate through each H (each for one basis function)
    for base_idx in range(H.shape[0]):

        # Get top num_chans channels by weight in original order
        top_chans = np.sort(
            np.argpartition(W[:, base_idx], -num_chans)[-num_chans:]
        )

        # Find peaks that are at least max_spike_freq seconds apart
        peak_indices, peak_heights = find_valid_peaks(
            H[base_idx], H_freq, max_spike_freq, height='auto'
        )

        # Remove border spikes: i.e., those in first/last context seconds (default: 5s)
        peak_indices, peak_heights = remove_border_spikes(
            peak_indices,
            peak_heights,
            H_freq,
            context,
            H.shape[1],
        )

        # Get a range of thresholds that are equally log-spaced
        thresholds = get_thresholds(
            peak_heights, spikes_per_cluster, log_transform=True
        )

        # Iterate through potential thresholds
        for threshold in thresholds:

            # Find index of spike with score closest to threshold, whilst exceeding threshold
            diff = peak_heights - threshold
            mask = np.where(diff > 0, diff, np.inf)
            index = np.argmin(mask)
            spike_idx = peak_indices[index]

            # Get peak in seconds
            spike_second = spike_idx / H_freq

            # Return data centred around (potential) spike
            spike_eeg = original_data.get_data(
                picks=top_chans,
                tmin=spike_second - context,
                tmax=spike_second + context,
            )

            # Get corresponding activation signal
            spike_activation = H[
                base_idx,
                spike_idx
                - context * H_freq : spike_idx
                + context * H_freq,
            ]

            # Resample to EEG frequency
            spike_activation = resample(spike_activation, spike_eeg.shape[1])

            # Add onset time to list
            onsets.append(len(onsets) * (context * 2) + context)

            # Add description of current spike: activation score and basis function index to list
            descriptions.append(
                f"Score: {peak_heights[index]}, Basis: {base_idx}"
            )

            # Add 21 channel signal, corresponding to activation & EEG, to list for all spikes across all bases
            all_potential_spikes.append(
                np.vstack([spike_activation, spike_eeg])
            )

    # Convert from list of each spike to continuous ndarray
    all_potential_spikes = np.hstack(all_potential_spikes)

    # Multiply activation such that it has same mean as EEG channels
    all_potential_spikes[0] *= np.mean(
        np.abs(all_potential_spikes[1:])
    ) / np.mean(all_potential_spikes[0])

    # Create info for new file: channels are integers up to num_chans
    info = mne.create_info(
        ch_names=list(map(str, range(num_chans + 1))),
        sfreq=original_data.info["sfreq"],
    )

    # Create RawArray from NumPy array and info
    raw = mne.io.RawArray(all_potential_spikes, info)

    # Create annotations from spike onsets (5s, 15s, ...) and descriptions that contain scores
    annotations = mne.Annotations(onsets, 0, descriptions)

    # Add annotations to RawArray
    raw.set_annotations(annotations)

    # Export as EDF
    mne.export.export_raw(save_location, raw, overwrite=True)


@task
def save_model(model: np.ndarray, save_path: str):
    """Save model (i.e., weights) to a specified location

    Parameters
    ----------
    model : np.ndarray
    save_path : str
    """
    joblib.dump(model, save_path)


@flow
def train(
    tmp_data_folder: str,
    raw_data_path: str,
):
    """Flow to train the model

    Parameters
    ----------
    tmp_data_folder : str
        The path to the folder where preprocessed data and the config file are stored.
    raw_data_path : str
        The path to the raw data file used for training the model.
    """

    # Config file is stored here
    config_path = Path(tmp_data_folder) / Path("detection_config.yaml")

    # Preprocessed data is stored here
    preprocessed_data_path = Path(tmp_data_folder) / Path("data_processed.pkl")

    # Load relevant config for preprocessing and model training
    preprocess_config = load_config(config_path)['preprocess']
    model_config = load_config(config_path)['model']

    # Model will be stored here - e.g., AA_04-07-2023_12:32
    model_dir = f'{Path(raw_data_path).parent / Path(raw_data_path).parent.stem}_{datetime.now().strftime("%d-%m-%Y_%H-%M")}'

    # Create temporary folder if it doesn't exist
    create_directory(model_dir)

    # Load preprocessed data
    preprocessed_data = get_preprocessed_data(preprocessed_data_path)

    # Train a model given the preprocessed data and model config
    W, H = train_model(model_config['rank'], preprocessed_data)

    # Load the original data
    original_data = get_raw_data(raw_data_path, preload=False)

    # Save proposed_ieds.edf for labelling
    save_spikes_for_labelling(
        original_data, W, H, max_spike_freq=preprocess_config['max_spike_freq'], save_location=Path(model_dir) / Path("proposed_ieds.edf")
    )

    # Save the model weights
    save_model(W, save_path=Path(model_dir) / Path("nmf_weights.pkl"))

    # Copy config file to model directory
    new_config_path = Path(model_dir) / Path("detection_config.yaml")
    copy_file(src=f'{config_path}', dst=f'{new_config_path}')

    # Copy thresholds file template to model directory
    new_thresholds_path = Path(model_dir) / Path("thresholds.yaml")
    copy_file(src='thresholds.yaml', dst=f'{new_thresholds_path}')
