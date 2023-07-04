"""Python script to train the model"""
import sys
from pathlib import Path

from detect_spikes import detect
from preprocess import preprocess


def detect_flow(
    raw_data_path,
    model_dir
):
    tmp_data_folder = preprocess(raw_data_path)

    save_path = str(Path(model_dir) / Path(raw_data_path).stem) + ".csv"
    print(save_path)
    detect(tmp_data_folder, model_dir, save_path)


if __name__ == "__main__":
    detect_flow(**dict(arg.split('=') for arg in sys.argv[1:]))
