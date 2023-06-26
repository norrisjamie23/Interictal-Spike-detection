"""
create Pydantic models
"""
from pydantic import BaseModel


class Location(BaseModel):
    """Specify the locations of inputs and outputs"""

    detection_config: str = "detection_config.yaml"
    detections: str = "data/processed/MS_detections.csv"
    thresholds: str = "thresholds.yaml"
    data_raw: str = "/Users/jamienorris/Documents/GOSH_SEEG_Analysis/stuart_data/MS/MS_IED_red_proc.edf"
    data_preprocess: str = "data/processed/MS.pkl"
    data_for_labelling: str = "data/processed/MS_spikes_for_labelling.edf"
    data_final: str = "data/final/predictions.pkl"
    model: str = "models/nmf_weights.pkl"
    input_notebook: str = "notebooks/analyze_results.ipynb"
    output_notebook: str = "notebooks/results.ipynb"
