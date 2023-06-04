"""Python script to train the model"""
import joblib
import nimfa
import numpy as np
import pandas as pd
from prefect import flow, task
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from config import Location, ModelParams


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
def train_model(model_params: ModelParams, ll_data: np.ndarray):
    """Train the model

    Parameters
    ----------
    model_params : ModelParams
        Parameters for the model
    ll_data : np.ndarray
        Line-length transformed data to use for training
    """

    rank = 5

    print("Training model")

    # 30 sets of randomly initialised BFs, 5 iterations, multiplicative updates
    nmf = nimfa.Nmf(ll_data, max_iter=5, rank=rank, n_run=30, objective="rss")
    nmf_fit = nmf()

    # Print RSS of best model
    print("RSS: %5.4f" % nmf_fit.fit.rss())

    # Get W and H to initialise next nmf (Alternating Nonnegative Least Squares Matrix Factorization)
    W = nmf_fit.basis()
    H = nmf_fit.coef()

    # Fit a single model initialising from best prior model, up to either 1000 iterations or min_residual of 1e-4
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
    print("RSS: %5.4f" % lsnmf_fit.fit.rss())

    W = np.array(lsnmf_fit.basis())
    H = np.array(lsnmf_fit.coef())

    # Below shouldn't be necessary, but keeping for now as I surely considered that last time
    # model = NMF(n_components=rank, max_iter=1, init='custom')

    # H = model.fit_transform(np.matmul(W, H).T, W=H.T.copy(order='C'), H=W.T.copy(order='C')).T
    # W = model.components_.T

    # # Reset max_iter, e.g., for inference
    # model.set_params(max_iter=1000)

    # np.save("W.npy", W)

    # return H matrix for training set
    return W


@task
def predict(grid: GridSearchCV, X_test: pd.DataFrame):
    """_summary_

    Parameters
    ----------
    grid : GridSearchCV
    X_test : pd.DataFrame
        Features for testing
    """
    return grid.predict(X_test)


@task
def save_model(model: GridSearchCV, save_path: str):
    """Save model to a specified location

    Parameters
    ----------
    model : GridSearchCV
    save_path : str
    """
    joblib.dump(model, save_path)


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
    svc_params: ModelParams = ModelParams(),
):
    """Flow to train the model

    Parameters
    ----------
    location : Location, optional
        Locations of inputs and outputs, by default Location()
    svc_params : ModelParams, optional
        Configurations for training the model, by default ModelParams()
    """
    data = get_processed_data(location.data_process)
    model = train_model(svc_params, data["ll_data"])
    predictions = predict(model, data["X_test"])
    save_model(model, save_path=location.model)
    save_predictions(predictions, save_path=location.data_final)


if __name__ == "__main__":
    train()
