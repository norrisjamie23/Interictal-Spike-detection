"""Python script to train the model"""
import joblib
import mne
import nimfa
import numpy as np
import pandas as pd
from prefect import flow, task

from config import Location, ModelParams, PreprocessParams
from utils import find_valid_peaks, get_raw_data


@task
def get_processed_data(data_location: str):
    """Get processed data from a specified location

    Parameters
    ----------
    data_location : str
        Location to get the data
    """
    return joblib.load(data_location)


@task
def train_model(nmf_params: ModelParams, ll_data: np.ndarray, rank: int = 5):
    """Train the model using NMF (Nonnegative Matrix Factorization)

    Parameters
    ----------
    model_params : ModelParams
        Parameters for the model
    ll_data : np.ndarray
        Line-length transformed data to use for training
    rank : int, optional
        Rank of the factorization (default is 5)

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
        ll_data, max_iter=5, rank=nmf_params.rank, n_run=30, objective="rss"
    )
    # nmf = nimfa.Nmf(
    #     ll_data, max_iter=1, rank=nmf_params.rank, n_run=1, objective="rss"
    # )
    # TODO enable above

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
        # max_iter=1,#000, TODO change to 1000
        min_residuals=1e-4,
    )
    lsnmf_fit = lsnmf()

    # Print RSS of final model
    print("Final model RSS: %5.4f" % lsnmf_fit.fit.rss())

    # Get weights (W) and activation scores (H)
    H = np.array(lsnmf_fit.basis())
    W = np.array(lsnmf_fit.coef())

    return W, H


@task
def save_spikes_to_label(
    original_data: mne.io.edf.edf.RawEDF,
    H: np.ndarray,
    W: np.ndarray,
    preprocess_params,
    num_chans: int = 20,
    spikes_per_cluster: int = 10,
    segment_length: int = 10,
    max_spike_freq=0.3,
):

    for base_idx in range(H.shape[0]):

        # Get top num_chans channels by weight in original order
        top_chans = np.sort(
            np.argpartition(W[:, base_idx], -num_chans)[-num_chans:]
        )

        # Find peaks that are at least max_spike_freq seconds apart
        peaks = find_valid_peaks(
            H[base_idx], preprocess_params.H_freq, max_spike_freq
        )

        top_chans
        peaks

        pass


# @task
# def predict(grid: GridSearchCV, X_test: pd.DataFrame):
#     """_summary_

#     Parameters
#     ----------
#     grid : GridSearchCV
#     X_test : pd.DataFrame
#         Features for testing
#     """
#     return grid.predict(X_test)
# @task
# def save_model(model: GridSearchCV, save_path: str):
#     """Save model to a specified location

#     Parameters
#     ----------
#     model : GridSearchCV
#     save_path : str
#     """
#     joblib.dump(model, save_path)


@task
def save_predictions(predictions: np.array, save_path: str):
    """Save predictions to a specified location

    Parameters
    ----------
    predictions : np.array
    save_path : str
    """
    joblib.dump(predictions, save_path)


@flow
def train(
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
    processed_data = get_processed_data(location.data_process)
    W, H = train_model(nmf_params, processed_data["ll_data"])

    original_data = get_raw_data(location.data_raw)
    save_spikes_to_label(original_data, W, H, preprocess_params)

    # predictions = predict(model, data["X_test"])
    # save_model(model, save_path=location.model)
    # save_predictions(predictions, save_path=location.data_final)


if __name__ == "__main__":
    train()
