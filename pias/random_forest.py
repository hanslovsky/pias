import threading

import numpy as np

from sklearn.ensemble import RandomForestClassifier

class ModelNotTrained(Exception):
    
    def __init__(self):
        super(ModelNotTrained, self).__init__("Model not trained yet")

class LabelsInconsistency(Exception):

    def __init__(self, required_labels, actual_labels):
        super(LabelsInconsistency, self).__init__('Need examples for %s but only got examples for %s', (required_labels, actual_labels))
        self.required_labels = required_labels
        self.actual_labels   = actual_labels

class RandomForestModelCache(object):

    def __init__(self, n_estimators=100, labels=(0,1), random_forest_kwargs=None):
        super(RandomForestModelCache, self).__init__()
        self.model                = None
        self.lock                 = threading.RLock()
        self.labels               = labels
        self.n_estimators         = n_estimators
        self.random_forest_kwargs = {} if random_forest_kwargs is None else random_forest_kwargs


    def train_model(self, samples, labels):

        if not np.all(np.unique(self.labels) == np.unique(labels)):
            raise LabelsInconsistency(self.labels, np.unique(labels))

        rf = RandomForestClassifier(n_estimators=self.n_estimators, **self.random_forest_kwargs)
        rf.fit(samples, labels)
        with self.lock:
            self.model = rf

        return rf

    def predict(self, samples):
        with self.lock:
            rf = self.model

        if rf is None:
            raise ModelNotTrained()

        return rf.predict_proba(samples)

    def get_model(self):
        with self.lock:
            return self.model

