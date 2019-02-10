from .ext import z5py

class EdgeFeatureIO(object):
    
    def __init__(self, container, edge_dataset='edges', edge_feature_dataset='edge_features'):
        super(EdgeFeatureIO, self).__init__()
        self.container            = container
        self.edge_dataset         = edge_dataset
        self.edge_feature_dataset = edge_feature_dataset

    def read(self):
        reader = z5py.File(self.container, use_zarr_format=False)
        return reader[self.edge_dataset][...], reader[self.edge_feature_dataset][...]

