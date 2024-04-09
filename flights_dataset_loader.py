import os
import urllib
import zipfile
import numpy as np
import torch
from torch_geometric.utils import dense_to_sparse
from torch_geometric_temporal.signal import StaticGraphTemporalSignal, DynamicGraphTemporalSignal

class FlightsDatasetLoader(object):

    def __init__(self, raw_data_dir=os.path.join(os.getcwd(), "data")):
        super(FlightsDatasetLoader, self).__init__()
        self.raw_data_dir = raw_data_dir
        self._load_data()
                
    def _load_data(self):

        X = np.load(os.path.join(self.raw_data_dir, "airport_data.npy"))
        X = X.astype(np.float32)

        # Normalise as in DCRNN paper (via Z-Score Method)
        means = np.nanmean(X, axis=(0, 2))
        X = X - means.reshape(1, -1, 1)
        stds = np.nanstd(X, axis=(0, 2))
        X = X / stds.reshape(1, -1, 1)
        
        self.A = torch.eye(X.shape[0])
        self.X = torch.from_numpy(X)
    
    def _get_edges_and_weights(self):
        edge_indices, values = dense_to_sparse(self.A)
        edge_indices = edge_indices.numpy()
        values = values.numpy()
        self.edges = edge_indices
        self.edge_weights = values
        
    def _generate_task(self, num_timesteps_in: int = 12, num_timesteps_out: int = 12):
       
        indices = [
            (i, i + (num_timesteps_in + num_timesteps_out))
            for i in range(self.X.shape[2] - (num_timesteps_in + num_timesteps_out) + 1)
        ]

        # Generate observations
        features, target = [], []
        for i, j in indices:
            features.append((self.X[:, :, i : i + num_timesteps_in]).numpy())
            target.append((self.X[:, 0, i + num_timesteps_in : j]).numpy())

        self.features = features
        self.targets = target

    # def get_dataset(
    #     self, num_timesteps_in: int = 12, num_timesteps_out: int = 12
    # ) -> StaticGraphTemporalSignal:
    #     self._get_edges_and_weights()
    #     self._generate_task(num_timesteps_in, num_timesteps_out)
    #     dataset = StaticGraphTemporalSignal(
    #         self.edges, self.edge_weights, self.features, self.targets
    #     )
    

    def get_dataset(
        self, num_timesteps_in: int = 12, num_timesteps_out: int = 12
    ) -> DynamicGraphTemporalSignal:
        # self._generate_task(num_timesteps_in, num_timesteps_out)
        
        list_of_edge_indices = self.edge_indices
        list_of_edge_weights = self.edge_weights
        
        dataset = DynamicGraphTemporalSignal(
            self.edge_indices,
            self.edge_weights, 
            self.features, 
            self.targets
        )

        return dataset
