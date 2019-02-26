import logging
import nifty
import threading

from .edges import EdgeFeatureIO

class EdgeFeatureCache(object):
    
    def __init__(self, container, edge_dataset, edge_feature_dataset):
        super(EdgeFeatureCache, self).__init__()
        self.logger = logging.getLogger('{}.{}'.format(self.__module__, type(self).__name__))
        self.logger.debug('Instantiating workflow with arguments %s', (container, edge_dataset, edge_feature_dataset))
        self.feature_io         = EdgeFeatureIO(container=container, edge_dataset=edge_dataset, edge_feature_dataset=edge_feature_dataset)
        self.edges              = None
        self.edge_features      = None
        self.edge_index_mapping = None
        self.graph              = None
        self.lock               = threading.RLock()

        self.update_edge_features()

    def get_edges_and_features(self):
        with self.lock:
            return self.edges, self.edge_features, self.edge_index_mapping, self.graph

    def update_edge_features(self):
        edges, features    = self.feature_io.read()
        edge_index_mapping = {(e[0], e[1]): index for index, e in enumerate(edges)}
        max_id = edges.max().item()
        graph = nifty.graph.UndirectedGraph(max_id + 1)
        graph.insertEdges(edges)
        with self.lock:
            self.edges              = edges
            self.edge_features      = features
            self.edge_index_mapping = edge_index_mapping
            self.graph              = graph
            return self.get_edges_and_features()