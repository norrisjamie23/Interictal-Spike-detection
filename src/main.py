"""
Create an iris flow
"""
from prefect import flow

from config import Location, ModelParams, PreprocessConfig
from preprocess import preprocess
from run_notebook import run_notebook
from train_model import train


@flow
def spike_detection_flow(
    location: Location = Location(),
    preprocess_config: PreprocessConfig = PreprocessConfig(),
    model_params: ModelParams = ModelParams(),
):
    """Flow to run the process, train, and run_notebook flows

    Parameters
    ----------
    location : Location, optional
        Locations of inputs and outputs, by default Location()
    process_config : ProcessConfig, optional
        Configurations for processing data, by default ProcessConfig()
    model_params : ModelParams, optional
        Configurations for training models, by default ModelParams()
    """
    preprocess(location, preprocess_config)
    train(location, model_params)
    run_notebook(location)


if __name__ == "__main__":
    spike_detection_flow()
