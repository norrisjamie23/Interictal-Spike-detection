# Interictal Spike detection

Python implementation of [Unsupervised Learning of Spatiotemporal Interictal Discharges in Focal Epilepsy](https://journals.lww.com/neurosurgery/abstract/2018/10000/unsupervised_learning_of_spatiotemporal_interictal.11.aspx) (Baud et al., 2017). Intracranial EEG data is preprocessed and the line-length is calculated for each channel. Non-negative matrix factorization (NNMF) is used to decompose this line-length transformed data into basis functions. The original paper determined the number of basis functions to use and calculated a threshold for a single basis function, allowing for fully-automated IED detection. We instead use a small amount of manual labelling to determine an optimal threshold for multiple basis functions.

### Setup 
All the necessary imports are in requirements.txt:
```
pip install -r requirements.txt
```

### Usage
1. Firstly, choose your parameters in *detection_config.yaml*. These are the default parameters:
    ```
    preprocess:
    highpass_freq: 0.1
    lowpass_freq: 50
    powerline_freq: 50
    max_spike_freq: 0.3

    model:
    rank: 5
    ```
    This allows us to specify our high- and low-pass filters, as well as a notch filter (e.g., 50Hz for UK and 60Hz for US). *max_spike_freq* determines the maximum frequency of IEDs: here we state that there should be at least 300ms between consecutive detections. Finally, we state that our NNMF model should have a rank of 5.

2. Train a model with:
    ```
    make train raw_data_path="/path/to/file.edf"
    ```
    This will create a subdirectory within the same directory as the edf file, e.g., /path/to/model. This subdirectory will have a name that includes the parent directory name, as well as the current date and time. E.g., Patient_A_10-07-2023_15-00:
    ```
    └── Patient_A_10-07-2023_15-00
        ├── detection_config.yaml
        ├── proposed_ieds.edf
        ├── thresholds.yaml
        └── nmf_weights.pkl
     ```
    * *detection_config.yaml* is a copy of the original file. It ensures you use the same parameters for applying the model as you did for training it.
    * *proposed_ieds.edf* is a file with some example IEDs from each basis function. It contains 50x 10 second epochs. Each of these is centered on a potential IED as determined by NNMF. For each basis function, you will need to determine if it's a valid or noise: if it's noise, leave the corresponding element in *thresholds.yaml* as _None_. Otherwise, choose a threshold and set the value to this.
    * *thresholds.yaml* is a file containing the thresholds. Its default is:
        ```
        thresholds: [null, null, null, null, null]
        ```
        You will need to make the amendments mentioned in the previous bullet point. For an example, you may change it to:
        ```
        thresholds: [0.05, 0.1, null, null, null]
        ```
        This would suggest that the first two basis functions have thresholds of 0.05 and 0.1 respectively, and the remainder are non-IED.
    * *nmf_weights.pkl* are the NNMF weights. You can ignore these.
3. Apply your trained model with:
    ```
    make detect raw_data_path="/path/to/file.edf" model_dir="/path/to/model"
    ```
    Note here that the former doesn't need to be the same file that the model was trained on - it just needs to be for the same patient and with the same channels - and the latter is simply the path to the directory created in the previous step. This will create a detections subdirectory in the same directory as the data you're running detection on. Within this subdirectory will be another subdirectory named after the model you've used, and within here will be a CSV with your detections. This will have 2 rows: the first will correspond to timings of each IED, and the second will have an index of which basis function it belongs to. These will be in chronological order.