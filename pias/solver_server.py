import logging
import struct
import time

import zmq

from .workflow import Workflow
from .server   import PublishSocket, ReplySocket, Server
from .zmq_util import send_int, recv_int, send_ints_multipart, recv_ints_multipart, send_more_int, _ndarray_as_bytes, _bytes_as_edges, send_ints

_EDGE_DATASET         = 'edges'
_EDGE_FEATURE_DATASET = 'edge-features'

_SUCCESS               = 0
_NO_SOLUTION_AVAILABLE = 1

_SET_EDGE_REP_SUCCESS           = 0
_SET_EDGE_REP_DO_NOT_UNDERSTAND = 1
_SET_EDGE_REP_EXCEPTION         = 2

_SET_EDGE_REQ_EDGE_LIST         = 0

_SOLUTION_UPDATE_REQUEST_RECEIVED = 0


class SolverServer(object):

    @staticmethod
    def default_edge_dataset():
        return _EDGE_DATASET

    @staticmethod
    def default_edge_feature_dataset():
        return _EDGE_FEATURE_DATASET
    
    def __init__(
            self,
            address_base,
            edge_n5_container,
            next_solution_id = 0,
            io_threads=1,
            edge_dataset=_EDGE_DATASET,
            edge_feature_dataset=_EDGE_FEATURE_DATASET):
        super(SolverServer, self).__init__()
        self.address_base = address_base
        self.logger = logging.getLogger('{}.{}'.format(self.__module__, type(self).__name__))
        self.logger.debug('Initializing workflow')
        self.workflow = Workflow(
            next_solution_id=next_solution_id, # TODO read from project file
            edge_n5_container=edge_n5_container,
            edge_dataset=edge_dataset,
            edge_feature_dataset=edge_feature_dataset)
        self.logger.debug('Initialized workflow')

        def current_solution(_, socket):
            solution = self.workflow.get_latest_state()
            if solution is None or solution.solution is None:
                send_int(socket, _NO_SOLUTION_AVAILABLE)
            else:
                send_more_int(socket, _SUCCESS)
                socket.send(_ndarray_as_bytes(solution.solution))

        def set_edge_labels_receive(socket):
            method = recv_int(socket)
            bytez  = socket.recv()
            return method, bytez

        def set_edge_labels_send(message, socket):
            self.logger.debug('Message is %s', message)
            method = message[0]
            self.logger.debug('Method is %s', method)
            try:
                if method == _SET_EDGE_REQ_EDGE_LIST:
                    labels = _bytes_as_edges(message[1])
                    self.logger.debug('Labels are %s', labels)
                    self.workflow.request_set_edge_labels(tuple((e[0], e[1]) for e in labels), tuple(e[2] for e in labels))
                    send_ints_multipart(socket, _SET_EDGE_REP_SUCCESS, len(labels))
                else:
                    send_ints_multipart(socket, _SET_EDGE_REP_DO_NOT_UNDERSTAND, method)
            except Exception as e:
                self.logger.debug('Sending exception `%s\' (%s)', e, type(e))
                send_more_int(socket, _SET_EDGE_REP_EXCEPTION)
                socket.send_string(str(e))

        def update_request_received_confirmation(_, socket):
            next_solution_id = self.workflow.request_update_state()
            send_ints_multipart(socket, _SOLUTION_UPDATE_REQUEST_RECEIVED, next_solution_id)

        def publish_new_solution(socket, message):
            self.logger.debug('Publishing new solution %s', message)
            send_ints(socket, *message)


        self.ping_address                    = '%s-ping' % address_base
        self.current_solution_address        = '%s-current-solution' % address_base
        self.set_edge_labels_address         = '%s-set-edge-labels' % address_base
        self.solution_update_request_address = '%s-update-solution' % address_base
        self.new_solution_address            = '%s-new-solution' % address_base

        ping_socket                    = ReplySocket(self.ping_address, timeout=10)
        solution_notifier_socket       = PublishSocket(self.new_solution_address, timeout=10 / 1000, send=publish_new_solution) # queue timeout is specified in seconds
        solution_request_socket        = ReplySocket(self.current_solution_address, timeout=10, respond=current_solution)
        solution_update_request_socket = ReplySocket(self.solution_update_request_address, timeout=10, respond=update_request_received_confirmation)
        set_edge_labels_request_socket = ReplySocket(self.set_edge_labels_address, timeout=10, respond=set_edge_labels_send, receive=set_edge_labels_receive)

        self.workflow.add_solution_update_listener(lambda solution_id, exit_code, solution: solution_notifier_socket.queue.put((solution_id, exit_code)))


        self.context = zmq.Context(io_threads=io_threads)
        self.server  = Server(
            ping_socket,
            solution_notifier_socket,
            solution_request_socket,
            set_edge_labels_request_socket,
            solution_update_request_socket)

        logging.info('Starting solver server at base address %s', address_base)
        logging.debug('Soluition notifier socket publishes at %s', self.new_solution_address)

        self.server.start(context=self.context)

        logging.info('Ping server at address %s', self.ping_address)

    def get_ping_address(self):
        return self.ping_address

    def get_current_solution_address(self):
        return self.current_solution_address

    def get_edge_labels_address(self):
        return self.set_edge_labels_address

    def get_solution_update_request_address(self):
        return self.solution_update_request_address

    def get_new_solution_address(self):
        return self.new_solution_address

    def shutdown(self):
        # TODO handle things like saving etc in here
        self.logger.debug('Shutting down server at base address %s', self.address_base)
        self.server.stop()
        self.workflow.stop()
