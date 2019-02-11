from __future__ import print_function

import contextlib
import numpy as np
import os
import shutil
import tempfile
import z5py

import unittest

import zmq

from pias import Workflow, SolverServer


@contextlib.contextmanager
def _tempdir():
    """A context manager for creating and then deleting a temporary directory."""
    tmpdir = tempfile.mkdtemp()
    try:
        yield tmpdir
    finally:
        shutil.rmtree(tmpdir)

def _mk_dummy_edge_data(
        container,
        edge_dataset=SolverServer.default_edge_dataset(),
        edge_feature_dataset=SolverServer.default_edge_feature_dataset()):

    edges = np.array(
        [[0, 1],
         [1, 2],
         [0, 2]],
        dtype=np.uint64)

    features = np.array(
        [[0.5, 1.0, 0.5],
         [0.7, 0.9, 0.8],
         [0.3, 0.1, 0.2]])

    f = z5py.File(container, 'w', use_zarr_format=False)
    f.create_dataset(edge_dataset, data=edges, dtype=np.uint64)
    f.create_dataset(edge_feature_dataset, data=features)


class TestSolverServerPing(unittest.TestCase):

    def test(self):

        with _tempdir() as tmpdir:
            address_base = 'inproc://address'
            container    = os.path.join(tmpdir, 'edge-group')
            _mk_dummy_edge_data(container)
            server = SolverServer(
                address_base=address_base,
                edge_n5_container=container)

            ping_socket = server.context.socket(zmq.REQ)
            ping_socket.setsockopt(zmq.SNDTIMEO, 30)
            ping_socket.setsockopt(zmq.RCVTIMEO, 30)
            ping_socket.connect(server.get_ping_address())

            # test ping three times
            for i in range(3):
                ping_socket.send_string('')
                ping_response = ping_socket.recv_string()
                self.assertEqual('', ping_response)

            server.shutdown()