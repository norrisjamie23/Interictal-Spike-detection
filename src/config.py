"""
create Pydantic models
"""
from pydantic import BaseModel


class Location(BaseModel):
    """Specify the locations of inputs and outputs"""

    # data_raw: str = "data/raw/iris.csv"
    data_raw: str = "/Users/jamienorris/Documents/GOSH_SEEG_Analysis/stuart_data/AJD_1097824/Interictal_AJD_red_proc.edf"
    data_preprocess: str = "data/processed/AJD.pkl"
    data_for_labelling: str = "data/processed/spikes_for_labelling.edf"
    data_final: str = "data/final/predictions.pkl"
    model: str = "models/nmf_weights.pkl"
    input_notebook: str = "notebooks/analyze_results.ipynb"
    output_notebook: str = "notebooks/results.ipynb"


class PreprocessParams(BaseModel):
    """Specify the parameters of the `process` flow"""

    highpass_freq = 50
    H_freq: int = 50


class ModelParams(BaseModel):
    """Specify the parameters of the `train` flow"""

    rank: int = 5
