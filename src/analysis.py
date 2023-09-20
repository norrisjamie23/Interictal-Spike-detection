import matplotlib.pyplot as plt
import numpy as np
from detect_spikes import load_model
from utils import get_raw_data
import matplotlib
from prefect import flow

# Configure global settings for matplotlib
matplotlib.rcParams['figure.figsize'] = (20, 10)
matplotlib.rcParams.update({'font.size': 12})

class Heatmap:
    def __init__(self, data_path, weights_path=None, weights_array=None):
        """
        Initialize Heatmap instance and populate the weights_reshaped array.

        The constructor initializes the object and calls the populate_weights_reshaped() 
        method to ensure the object is in a 'ready-to-use' state immediately after instantiation.

        You can either provide a 'weights_path' to load the pre-trained model weights or directly
        provide a 'weights_array'.

        Parameters:
        data_path: str, Path to the raw EEG data.
        weights_path: str, optional, Path to the pre-trained model weights.
        weights_array: numpy.ndarray, optional, Directly provide the weights as an array.
        """
        
        if weights_array is not None:
            self.weights = weights_array
        elif weights_path is not None:
            self.weights = load_model(weights_path)
        else:
            raise ValueError("Either 'weights_path' or 'weights_array' must be provided.")

        self.channels = get_raw_data(data_path, preload=False).info['ch_names']
        self.populate_weights_reshaped()

    def populate_weights_reshaped(self):
        """
        Populate a 3D tensor that reshapes the 2D weight matrix.
        
        This method reshapes the weights to align them with the layout of the SEEG electrodes.
        """
        # Create an empty 3D array to store the reshaped weights
        self.weights_reshaped = np.zeros((self.weights.shape[1], len(set([channel[0] for channel in self.channels])), max([int(channel[1:]) for channel in self.channels])))
        
        # Sort electrodes alphabetically
        electrodes = sorted(list(set([channel[0] for channel in self.channels])))

        # Populate the reshaped array
        for base in range(self.weights_reshaped.shape[0]):
            for row in range(self.weights_reshaped.shape[1]):
                for col in range(self.weights_reshaped.shape[2]):
                    contact_name = electrodes[row] + str(col + 1)
                    if contact_name in self.channels:
                        self.weights_reshaped[base, row, col] = self.weights[self.channels.index(contact_name)][base] + 1e-10

    def plot_IED_heatmap(self, basis_index=0, cmap="winter", x_label='', y_label='', title='', save_path=None):
        """
        Plot a heatmap visualizing different interictal epileptiform activity subgroups.

        Parameters:
        basis_index: int, Index of the basis function in W.
        cmap: str, Colormap name.
        x_label: str, Label for the x-axis.
        y_label: str, Label for the y-axis.
        title: str, Title of the plot.
        save_path: str, optional, Path to save the generated plot. If None, the plot is not saved.
        """
        # Number of unique electrodes and maximum number of contacts per electrode
        num_electrodes = len(set([channel[0] for channel in self.channels]))
        max_electrode_channels = max([int(channel[1:]) for channel in self.channels])
        
        # Sort electrodes alphabetically for labeling
        sorted_electrodes = sorted(list(set([channel[0] for channel in self.channels])))
        
        # Initialize the plotting grid
        fig, axes = plt.subplots(nrows=num_electrodes, sharex=True)
        
        for row_idx, weights_row in enumerate(self.weights_reshaped[basis_index]):
            # Mask negligible weights
            weights_row = np.ma.masked_where(weights_row < 1e-10, weights_row)
            
            # Generate heatmap row
            color_map = axes[row_idx].pcolor(
                np.expand_dims(weights_row, axis=0),
                edgecolors="k",
                linewidths=1,
                cmap=cmap,
                vmin=0,
                vmax=self.weights[:, basis_index].max()
            )
            
            # Add label for the electrode
            axes[row_idx].set_ylabel(sorted_electrodes[row_idx], rotation=0)
            axes[row_idx].get_yaxis().set_label_coords(-0.015, 0.3)
            
            # Remove y and x ticks
            axes[row_idx].set_yticks([])
            axes[row_idx].tick_params(axis="x", length=0)

        # Add colorbar
        fig.colorbar(color_map, ax=axes)
        
        # Add x ticks for the channels
        plt.xticks(np.arange(0.5, max_electrode_channels + 0.5, 1.0), range(1, max_electrode_channels + 1))
        
        # Add additional labels and title if provided
        if x_label:
            fig.text(0.44, 0.045, x_label, ha='center', fontsize=16)
        if y_label:
            fig.text(0.093, 0.5, y_label, va='center', rotation='vertical', fontsize=16)
        if title:
            plt.suptitle(title, x=0.44, y=0.925, fontsize=18)

        # Save the plot if save_path is provided
        if save_path:
            plt.savefig(save_path)
        
        plt.show()


@flow
def plot_IED_heatmap(weights_path, data_path, basis_index=0, save_path=None):

    importances_heatmap = Heatmap(data_path=data_path, weights_path=weights_path)

    # Update the title to reflect the basis index
    title = f'Distribution of Weights Across Channels for Basis Function {basis_index}'
    
    importances_heatmap.plot_IED_heatmap(basis_index, cmap="viridis",
                                        x_label='Channel within electrode (higher values are deeper in brain)', 
                                        y_label='Electrode',
                                        title=title,
                                        save_path=save_path)
