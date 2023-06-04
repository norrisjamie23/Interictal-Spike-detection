"""Python script to train the model"""
import joblib
import mne
import nimfa
import numpy as np
import pandas as pd
from prefect import flow, task
from scipy.signal import savgol_filter

from config import Location, ModelParams

# from utils import get_raw_data


def get_minimum_thresh(H, k=1):
    """Compute the threshold to use for scipy.signal.find_peaks

    Parameters
    ----------
    H : array-like
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
def save_spikes_to_label(
    original_data: mne.io.edf.edf.RawEDF,
    H: np.ndarray,
    W: np.ndarray,
    num_chans: int = 20,
    percentile_increments: int = 10,
):

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


@task
def get_raw_data(data_location: str):
    """Read raw data

    Parameters
    ----------
    data_location : str
        The location of the raw data
    """
    return mne.io.read_raw_edf(data_location, preload=True)


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
    save_spikes_to_label(original_data, W, H)

    # predictions = predict(model, data["X_test"])
    # save_model(model, save_path=location.model)
    # save_predictions(predictions, save_path=location.data_final)


if __name__ == "__main__":
    train()
