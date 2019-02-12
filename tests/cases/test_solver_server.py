from __future__ import print_function

import logging
import time

_logger = logging.getLogger(__name__)

import contextlib
import os
import shutil
import tempfile
import unittest

import numpy as np
import z5py
import zmq

from pias import SolverServer
from pias.solver_server import _bytes_as_int, _int_as_bytes, _edges_as_bytes, _NO_SOLUTION_AVAILABLE, _SET_EDGE_REQ_EDGE_LIST, _SET_EDGE_REP_SUCCESS, _SET_EDGE_REP_DO_NOT_UNDERSTAND


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
         [0, 2],
         [1, 3],
         [2, 3]],
        dtype=np.uint64)

    features = np.array(
        [[0.5, 1.0, 0.5],
         [0.7, 0.9, 0.8],
         [0.3, 0.9, 0.2],
         [0.5, 0.2, 0.6],
         [0.4, 0.1, 0.3]])

    labels = (0, 0, 0, 1, 1)

    f = z5py.File(container, 'w', use_zarr_format=False)
    f.create_dataset(edge_dataset, data=edges, dtype=np.uint64)
    f.create_dataset(edge_feature_dataset, data=features)

    return edges, features, labels


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

class TestSolverCurrentSolution(unittest.TestCase):

    def test(self):

        with _tempdir() as tmpdir:
            address_base = 'inproc://address'
            container    = os.path.join(tmpdir, 'edge-group')
            _mk_dummy_edge_data(container)
            server = SolverServer(
                address_base=address_base,
                edge_n5_container=container)

            current_solution_socket = server.context.socket(zmq.REQ)
            current_solution_socket.setsockopt(zmq.SNDTIMEO, 30)
            current_solution_socket.setsockopt(zmq.RCVTIMEO, 30)
            current_solution_socket.connect(server.get_current_solution_address())
            current_solution_socket.send_string('')
            solution_bytes = current_solution_socket.recv()
            _logger.debug('%d solution bytes: %s', len(solution_bytes), solution_bytes)
            solution = _bytes_as_int(solution_bytes)
            self.assertEqual(1, len(solution))
            self.assertEqual(_NO_SOLUTION_AVAILABLE, solution[0])

            server.shutdown()

class TestSolverSetEdgeLabels(unittest.TestCase):

    def test(self):

        with _tempdir() as tmpdir:
            address_base = 'inproc://address'
            container    = os.path.join(tmpdir, 'edge-group')
            edges, _, labels = _mk_dummy_edge_data(container)
            server = SolverServer(
                address_base=address_base,
                edge_n5_container=container)

            edge_label_socket = server.context.socket(zmq.REQ)
            edge_label_socket.setsockopt(zmq.SNDTIMEO, 30)
            edge_label_socket.setsockopt(zmq.RCVTIMEO, 30)
            edge_label_socket.connect(server.get_edge_labels_address())
            edge_label_socket.send(_int_as_bytes(_SET_EDGE_REQ_EDGE_LIST), flags=zmq.SNDMORE)
            edge       = (edges[0, 0].item(), edges[0, 1].item(), labels[0])
            edge_bytes = _edges_as_bytes((edge,))
            edge_label_socket.send(edge_bytes)
            response_code = _bytes_as_int(edge_label_socket.recv())
            self.assertEqual(1, len(response_code))
            self.assertEqual(_SET_EDGE_REP_SUCCESS , response_code[0])
            num_edges = _bytes_as_int(edge_label_socket.recv())
            self.assertEqual(1, len(num_edges))
            self.assertEqual(1, num_edges[0])

            edges_as_tuples = tuple((e[0].item(), e[1].item(), l) for e, l in zip(edges, labels))
            edge_label_socket.send_multipart(msg_parts=(_int_as_bytes(_SET_EDGE_REQ_EDGE_LIST), _edges_as_bytes(edges_as_tuples)))
            response_code = _bytes_as_int(edge_label_socket.recv())
            self.assertEqual(1, len(response_code))
            self.assertEqual(_SET_EDGE_REP_SUCCESS, response_code[0])
            num_edges = _bytes_as_int(edge_label_socket.recv())
            self.assertEqual(1, len(num_edges))
            self.assertEqual(len(labels), num_edges[0])

            edge_label_socket.send_multipart(msg_parts=(_int_as_bytes(-1), b''))
            response_code = _bytes_as_int(edge_label_socket.recv())
            self.assertEqual(1, len(response_code))
            self.assertEqual(_SET_EDGE_REP_DO_NOT_UNDERSTAND, response_code[0])
            message = _bytes_as_int(edge_label_socket.recv())
            self.assertEqual(1, len(message))
            self.assertEqual(-1, message[0])

            server.shutdown()