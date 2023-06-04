"""Python script to process the data"""

import mne
from prefect import flow, task


@task
def get_raw_data(data_location: str):
    """Read raw data

    Parameters
    ----------
    data_location : str
        The location of the raw data
    """
    return mne.io.read_raw_edf(data_location, preload=True)
