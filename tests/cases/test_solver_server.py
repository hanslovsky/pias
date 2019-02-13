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
from pias.solver_server import _NO_SOLUTION_AVAILABLE, _SET_EDGE_REQ_EDGE_LIST, _SET_EDGE_REP_SUCCESS, _SET_EDGE_REP_DO_NOT_UNDERSTAND, _SET_EDGE_REP_EXCEPTION
from pias import zmq_util


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
            solution = zmq_util.recv_int(current_solution_socket)
            self.assertEqual(_NO_SOLUTION_AVAILABLE, solution)

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
            edge       = (edges[0, 0].item(), edges[0, 1].item(), labels[0])
            zmq_util.send_more_int(edge_label_socket, _SET_EDGE_REQ_EDGE_LIST)
            edge_label_socket.send(zmq_util._edges_as_bytes((edge,)))
            response_code = zmq_util.recv_int(edge_label_socket)
            self.assertEqual(_SET_EDGE_REP_SUCCESS , response_code)
            num_edges = zmq_util.recv_int(edge_label_socket)
            self.assertEqual(1, num_edges)

            edges_as_tuples = tuple((e[0].item(), e[1].item(), l) for e, l in zip(edges, labels))
            zmq_util.send_more_int(edge_label_socket, _SET_EDGE_REQ_EDGE_LIST)
            edge_label_socket.send(zmq_util._edges_as_bytes(edges_as_tuples))
            response_code = zmq_util.recv_int(edge_label_socket)
            self.assertEqual(_SET_EDGE_REP_SUCCESS, response_code)
            num_edges = zmq_util.recv_int(edge_label_socket)
            self.assertEqual(len(labels), num_edges)

            zmq_util.send_ints_multipart(edge_label_socket, -1, 0)
            response_code, message_type = zmq_util.recv_ints_multipart(edge_label_socket)
            self.assertEqual(_SET_EDGE_REP_DO_NOT_UNDERSTAND, response_code)
            self.assertEqual(-1, message_type)

            zmq_util.send_more_int(edge_label_socket, _SET_EDGE_REQ_EDGE_LIST)
            edge_label_socket.send(bytearray(8))
            response_code = zmq_util.recv_int(edge_label_socket)
            self.assertEqual(_SET_EDGE_REP_EXCEPTION, response_code)
            exception = edge_label_socket.recv_string()


            server.shutdown()