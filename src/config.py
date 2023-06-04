"""
create Pydantic models
"""
from typing import List

from pydantic import BaseModel, validator


def must_be_non_negative(v: float) -> float:
    """Check if the v is non-negative

    Parameters
    ----------
    v : float
        value

    Returns
    -------
    float
        v

    Raises
    ------
    ValueError
        Raises error when v is negative
    """
    if v < 0:
        raise ValueError(f"{v} must be non-negative")
    return v


class Location(BaseModel):
    """Specify the locations of inputs and outputs"""

    # data_raw: str = "data/raw/iris.csv"
    data_raw: str = "/Users/jamienorris/Documents/GOSH_SEEG_Analysis/stuart_data/AJD_1097824/Interictal_AJD_red_proc.edf"
    data_process: str = "data/processed/AJD.pkl"
    data_final: str = "data/final/predictions.pkl"
    model: str = "models/svc.pkl"
    input_notebook: str = "notebooks/analyze_results.ipynb"
    output_notebook: str = "notebooks/results.ipynb"


class ProcessConfig(BaseModel):
    """Specify the parameters of the `process` flow"""

    highpass_freq = 50
    H_freq: int = 50
    rank: int = 5

    _validated_test_size = validator("test_size", allow_reuse=True)(
        must_be_non_negative
    )


class ModelParams(BaseModel):
    """Specify the parameters of the `train` flow"""

    rank: int = 5

    _validated_fields = validator("*", allow_reuse=True, each_item=True)(
        must_be_non_negative
    )
