import logging

import numpy as np
import threading


class EdgeLabelCache(object):
    
    def __init__(self):
        super(EdgeLabelCache, self).__init__()
        self.logger = logging.getLogger('{}.{}'.format(self.__module__, type(self).__name__))
        self.edge_label_map     = {}
        self.edge_index_mapping = None
        self.lock               = threading.RLock()

    def update_labels(self, edges, labels):
        with self.lock:

            if self.edge_index_mapping is None:
                return

            for e, l in zip(edges, labels):
                if e not in self.edge_index_mapping:
                    self.logger.debug('Edge %s not in edge-index-mapping %s', e, self.edge_index_mapping)
                    continue
                index = self.edge_index_mapping[e]
                self.edge_label_map[index] = l

    def get_sample_and_label_arrays(self, samples):
        with self.lock:
            edge_indices = np.fromiter(self.edge_label_map.keys(), dtype=np.uint64)
            labels       = np.fromiter(self.edge_label_map.values(), dtype=np.uint64)
        return samples[edge_indices, ...], labels

    def update_edge_index_mapping(self, edge_index_mapping):
        with self.lock:
            self.logger.debug('Updating edge-index-mapping: %s', edge_index_mapping)
            self.edge_index_mapping = edge_index_mapping


