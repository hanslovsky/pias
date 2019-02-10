from __future__ import print_function

import contextlib
import numpy as np
import os
import shutil
import tempfile
import unittest

from pias import EdgeFeatureIO
from pias.ext import z5py


class TestEdgeIO(unittest.TestCase):

    @contextlib.contextmanager
    def _tempdir(self):
        """A context manager for creating and then deleting a temporary directory."""
        tmpdir = tempfile.mkdtemp()
        try:
            yield tmpdir
        finally:
            shutil.rmtree(tmpdir)

    def testReadEdgeFeatures(self):

        edges = np.array(
            [[0, 1],
             [1, 2],
             [0, 2]],
            dtype=np.uint64)

        features = np.array(
            [[0.5, 1.0, 0.5],
             [0.7, 0.9, 0.8],
             [0.3, 0.1, 0.2]])

        with self._tempdir() as tmpdir:
            container = os.path.join(tmpdir, 'some', 'feature', 'test')
            os.makedirs(container, exist_ok=True)

            f = z5py.File(container, 'w', use_zarr_format=False)
            f.create_dataset('edges', data=edges)
            f.create_dataset('edge_features', data=features)

            edgeIO = EdgeFeatureIO(container, edge_dataset='edges', edge_feature_dataset='edge_features')
            e, f = edgeIO.read()
            self.assertTrue(np.all(edges == e))
            self.assertTrue(np.all(features == f))