import logging
import struct

import zmq

from .workflow import Workflow
from .server   import PublishSocket, ReplySocket, Server

_EDGE_DATASET         = 'edges'
_EDGE_FEATURE_DATASET = 'edge-features'

_SUCCESS               = 0
_NO_SOLUTION_AVAILABLE = 1

# big-endian:
# https://docs.python.org/2/library/struct.html#byte-order-size-and-alignment
_USE_BIG_ENDIAN  = True
_ENDIANNESS      = '>' if _USE_BIG_ENDIAN else '<'
_INTEGER_PATTERN = f'{_ENDIANNESS}i'
_UINT64_PATTERN  = f'{_ENDIANNESS}Q'
_EDGE_PATTERN    = 'QQi'

_SET_EDGE_REP_SUCCESS           = 0
_SET_EDGE_REP_DO_NOT_UNDERSTAND = 1

_SET_EDGE_REQ_EDGE_LIST         = 0


def _int_as_bytes(number):
    return struct.pack(_INTEGER_PATTERN, number)

def _bytes_as_int(b):
    return struct.unpack(_INTEGER_PATTERN, b)

def _edges_as_bytes(edges):
    pattern = f'{_ENDIANNESS}%s' % (_EDGE_PATTERN * len(edges))
    return struct.pack(pattern, *tuple(e for edge in edges for e in edge))

def _bytes_as_edges(b):
    # 8 + 8 + 4
    # label1, label2, 1 or 0
    # uint64,uint64,byte
    entry_size = 20
    a = len(b)
    assert len(b) % entry_size == 0
    num_edges = len(b) // entry_size
    pattern = f'{_ENDIANNESS}%s' % (_EDGE_PATTERN * num_edges)
    e = struct.unpack(pattern, b)
    return tuple((e[i+0], e[i+1], e[i+2]) for i in range(0, len(e), 3))

def _ndarray_as_bytes(ndarray):
    # java always big endian
    # https://stackoverflow.com/questions/981549/javas-virtual-machines-endianness
    return (ndarray.byteswap() if _USE_BIG_ENDIAN else ndarray).tobytes()

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
            io_threads=1,
            edge_dataset=_EDGE_DATASET,
            edge_feature_dataset=_EDGE_FEATURE_DATASET):
        super(SolverServer, self).__init__()
        self.logger = logging.getLogger('{}.{}'.format(self.__module__, type(self).__name__))
        workflow = Workflow(
            edge_n5_container=edge_n5_container,
            edge_dataset=edge_dataset,
            edge_feature_dataset=edge_feature_dataset)

        def current_solution(_, socket):
            solution = workflow.get_solution()
            if solution is None:
                socket.send(_int_as_bytes(_NO_SOLUTION_AVAILABLE))
            else:
                socket.send_more(_int_as_bytes(_SUCCESS))
                socket.send(_ndarray_as_bytes(solution))

        def set_edge_labels_receive(socket):
            method = _bytes_as_int(socket.recv())
            bytez  = socket.recv()
            return method[0], bytez

        def set_edge_labels_send(message, socket):
            method = message[0]
            if method == _SET_EDGE_REQ_EDGE_LIST:
                labels = _bytes_as_edges(message[1])
                socket.send_multipart(msg_parts=(_int_as_bytes(_SET_EDGE_REP_SUCCESS), _int_as_bytes(len(labels))))
            else:
                socket.send_multipart(msg_parts=(_int_as_bytes(_SET_EDGE_REP_DO_NOT_UNDERSTAND), _int_as_bytes(method)))

        self.ping_address             = '%s-ping' % address_base
        self.current_solution_address = '%s-current-solution' % address_base
        self.set_edge_labels_address  = '%s-set-edge-labels' % address_base

        ping_socket                    = ReplySocket(self.ping_address, timeout=10)
        solution_notifier_socket       = PublishSocket('%s-new-solution' % address_base, timeout=10 / 1000, send=lambda socket, message: socket.sent_str(message)) # queue timeout is specified in seconds
        solution_request_socket        = ReplySocket(self.current_solution_address, timeout=10, respond=current_solution)
        # solution_update_request_socket = ReplySocket('%s-get-solution' % address_base, timeout=10, respond=update_request_received_confirmation)
        set_edge_labels_request_socket = ReplySocket(self.set_edge_labels_address, timeout=10, respond=set_edge_labels_send, receive=set_edge_labels_receive)

        workflow.add_solution_update_listener(lambda solution: solution_notifier_socket.queue.put(''))


        self.context = zmq.Context(io_threads=io_threads)
        self.server  = Server(
            ping_socket,
            solution_notifier_socket,
            solution_request_socket,
            set_edge_labels_request_socket)

        logging.info('Starting solver server at base address %s', address_base)

        self.server.start(context=self.context)

        logging.info('Ping server at address %s', self.ping_address)

    def get_ping_address(self):
        return self.ping_address

    def get_current_solution_address(self):
        return self.current_solution_address

    def get_edge_labels_address(self):
        return self.set_edge_labels_address

    def shutdown(self):
        # TODO handle things like saving etc in here
        self.server.stop()

