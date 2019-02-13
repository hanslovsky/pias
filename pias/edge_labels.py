import numpy as np
import threading


class EdgeLabelCache(object):
    
    def __init__(self):
        super(EdgeLabelCache, self).__init__()
        self.edge_label_map     = {}
        self.edge_index_mapping = None
        self.lock               = threading.RLock()
        self.edge_labels_dirty  = False

    def update_labels(self, edges, labels):
        with self.lock:

            if self.edge_index_mapping is None:
                return

            for e, l in zip(edges, labels):
                if e not in self.edge_index_mapping:
                    continue
                index = self.edge_index_mapping[e]
                self.edge_label_map[index] = l
            self.edge_labels_dirty = True

    def get_sample_and_label_arrays(self, samples):
        with self.lock:
            edge_indices = np.fromiter(self.edge_label_map.keys(), dtype=np.uint64)
            labels       = np.fromiter(self.edge_label_map.values(), dtype=np.uint64)
            is_dirty     = self.edge_labels_dirty
        return samples[edge_indices, ...], labels, is_dirty

    def clean_labels(self):
        with self.lock:
            self.edge_labels_dirty = False

    def update_edge_index_mapping(self, edge_index_mapping):
        with self.lock:
            self.edge_index_mapping = edge_index_mapping


