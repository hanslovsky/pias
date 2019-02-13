from __future__ import absolute_import

import logging
import threading

from .agglomeration_model import AgglomerationModelCache
from .edge_feature_cache import EdgeFeatureCache
from .edge_labels import  EdgeLabelCache
from .random_forest import LabelsInconsistency, ModelNotTrained, RandomForestModelCache

class Workflow(object):
    
    def __init__(
            self,
            edge_n5_container,
            edge_dataset,
            edge_feature_dataset,
            n_estimators=100,
            random_forest_kwargs=None):
        super(Workflow, self).__init__()
        self.logger = logging.getLogger('{}.{}'.format(self.__module__, type(self).__name__))
        self.logger.debug('Instantiating workflow with arguments %s', (edge_n5_container, edge_dataset, edge_feature_dataset, n_estimators, random_forest_kwargs))
        self.edge_feature_cache        = EdgeFeatureCache(edge_n5_container, edge_dataset=edge_dataset, edge_feature_dataset=edge_feature_dataset)
        self.edge_label_cache          = EdgeLabelCache()
        self.random_forest_model_cache = RandomForestModelCache(labels=(0, 1), n_estimators=n_estimators, random_forest_kwargs=random_forest_kwargs)
        self.agglomeration_model_cache = AgglomerationModelCache()
        # TODO do we need to lock in any place?
        self.lock                      = threading.RLock()

        self.solution_update_notify    = []

        self.update_edges()


    # TODO potential race conditions in methods below
    def update_edges(self):
        edges, edge_features, edge_index_mapping = self.edge_feature_cache.update_edge_features()
        self.edge_label_cache.update_edge_index_mapping(edge_index_mapping)
        self.agglomeration_model_cache.update_graph(edges=edges)
        try:
            weights = self.random_forest_model_cache.predict(edge_features)
        except ModelNotTrained:
            try:
                self.train_model(edge_features=edge_features)
                weights = self.random_forest_model_cache.predict(edge_features)
            except LabelsInconsistency:
                weights = None

        if weights is not None:
            self.agglomeration_model_cache.optimize(weights)


    def set_edge_labels(self, edges, labels):
        self.edge_label_cache.update_labels(edges, labels)

    def train_model(self, edge_features=None):
        if edge_features is None:
            _, edge_features, _ = self.edge_feature_cache.get_edges_and_features()
        samples, labels, _ = self.edge_label_cache.get_sample_and_label_arrays(edge_features)
        self.random_forest_model_cache.train_model(samples, labels)

    def get_solution(self):
        return self.agglomeration_model_cache.get_solution()

    def update_solution(self, weights):
        solution = self.agglomeration_model_cache.optimize(weights)
        for listener in self.solution_update_notify:
            listener(solution)


    def add_solution_update_listener(self, listener):
        self.solution_update_notify.append(listener)

