import json
import urllib
import numpy as np
from torch_geometric_temporal.signal.static_graph_temporal_signal import StaticGraphTemporalSignal

class FlightsLoader(object):
    
    def __init__(self):
        self._load_data()

    def _load_data(self):
        # load data from json file in the local repository
        with open("./data/flights.json") as f:
            self._dataset = json.load(f)
        
    def _get_edges(self):
        self._edges = np.array(self._dataset["edges"]).T

    def _get_edge_weights(self):
        self._edge_weights = np.ones(self._edges.shape[1])

    def _get_targets_and_features(self):
        stacked_target = np.array(self._dataset["FX"]) #, dtype=np.float64)
        # if the stacked_target contains null values, replace them with nan
        # stacked_target = np.where(stacked_target == None, np.nan, stacked_target)

        
        self.features = [
            stacked_target[i : i + self.lags, :].T
            for i in range(stacked_target.shape[0] - self.lags)
        ]
        self.targets = [
            stacked_target[i + self.lags, :].T
            for i in range(stacked_target.shape[0] - self.lags)
        ]

    def get_dataset(self, lags: int = 4) -> StaticGraphTemporalSignal:
       
        self.lags = lags
        self._get_edges()
        self._get_edge_weights()
        self._get_targets_and_features()
        dataset = StaticGraphTemporalSignal(
            self._edges, self._edge_weights, self.features, self.targets
        )
        return dataset