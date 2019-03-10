import os
import signal
import tempfile
import threading
from datetime import datetime

import zmq

from .ext import z5py
from .pias_logging import levels as log_levels
from .pias_logging import logging
from .server import PublishSocket, ReplySocket, Server
from .workflow import Workflow
from .zmq_util import send_int, recv_int, send_ints_multipart, send_more_int, _ndarray_as_bytes, _bytes_as_edges, \
    send_ints

_EDGE_DATASET         = 'edges'
_EDGE_FEATURE_DATASET = 'edge-features'
_PAINTERA_DATA_KEY    = 'painteraData'

_SUCCESS               = 0
_NO_SOLUTION_AVAILABLE = 1

_SET_EDGE_REP_SUCCESS           = 0
_SET_EDGE_REP_DO_NOT_UNDERSTAND = 1
_SET_EDGE_REP_EXCEPTION         = 2

_SET_EDGE_REQ_EDGE_LIST         = 0

_SOLUTION_UPDATE_REQUEST_RECEIVED = 0


API_RESPONSE_OK               = 0
API_RESPONSE_UNKNOWN_ERROR    = 1
API_RESPONSE_ENDPOINT_UNKNOWN = 2

API_RESPONSE_DATA_STRING  = 0
API_RESPONSE_DATA_BYTES   = 1
API_RESPONSE_DATA_INT     = 2
API_RESPONSE_DATA_UNKNOWN = 3

API_HELP_STRING_TEMPLATE = '''
Paintera Interactive Solver Server

Connect to {address_base} via a zmq.REP socket and access information via endpoints.
The /help endpoint sends  a single zmq string response. All endpoints under /api send at least two messages:
 - one integer (0 if endpoint is known, 1 if unknown error occurred during processing, 2 if endpoint is unknown)
 - one integer specifying the number of messages to be sent (might be 0)
 - optional (if number of messages is larger than 0) n times:
   - integer indicating type of message:
      - 0 string
      - 1 bytes
      - 2 integer
      - 3 unknown/structured (look at help if available, will be sent as bytes)
   - actual contents

`'
    REQ/REP Repsond empty string as pong
/
    REQ/REP Respond empty string as pong
/help
    REQ/REP Send this help message (single zmq string response)
/api/n5/container
    REQ/REP Path to n5 container holding paintera dataset with edges and features
/api/n5/dataset
    REQ/REP Path to paintera dataset in n5 container
/api/n5/all
    REQ/REP Send both container and dataset as multiple messages.
/api/save-ground-truth-labels
    REQ/REP Serialize current ground truth labels (uv-pairs and labels) into server directory

Use the following addresses for specific queries:

{ping_address}
    REQ/REP: Responds with empty string as pong
{current_solution_address}
    REQ/REP: Responds with current solution (if any)
{set_edge_labels_address}
    REQ/REP: Submit list of edge labels
{solution_update_request_address}
    PUB/SUB: Subscribe to `' (empty string) to be notified whenever a new solution is available
{api_endpoint_address}
    REQ/REP for api endpoints
'''


class SolverServer(object):

    @staticmethod
    def default_edge_dataset():
        return _EDGE_DATASET

    @staticmethod
    def default_edge_feature_dataset():
        return _EDGE_FEATURE_DATASET

    @staticmethod
    def is_paintera_data(container, dataset):
        with z5py.File(container, 'r') as f:
            return _PAINTERA_DATA_KEY in f[dataset].attrs

    @staticmethod
    def is_paintera_label_data(container, dataset):
        with z5py.File(container, 'r') as f:
            return f[dataset].attrs[_PAINTERA_DATA_KEY]['type'] == 'label'

    @staticmethod
    def ping_address(address_base):
        return '%s-ping' % address_base

    @staticmethod
    def current_solution_address(address_base):
        return '%s-current-solution' % address_base

    @staticmethod
    def set_edge_labels_address(address_base):
        return '%s-set-edge-labels' % address_base

    @staticmethod
    def solution_update_request_address(address_base):
        return '%s-update-solution' % address_base

    @staticmethod
    def new_solution_address(address_base):
        return '%s-new-solution' % address_base

    @staticmethod
    def api_endpoint_address(address_base):
        return address_base

    @staticmethod
    def api_endpoint_respond(socket, return_code, *messages):

        send_more_int(socket, return_code)
        send_int(socket, len(messages), flags=0 if len(messages) == 0 else zmq.SNDMORE)
        for index, (message_type, message) in enumerate(messages):
            send_more_int(socket, message_type)
            flag = 0 if index == len(messages) - 1 else zmq.SNDMORE
            if message_type == API_RESPONSE_DATA_STRING:
                socket.send_string(message, flags=flag)
            elif message_type == API_RESPONSE_DATA_BYTES or message_type == API_RESPONSE_DATA_UNKNOWN:
                socket.send(message, flags=flag)
            elif message_type == API_RESPONSE_DATA_INT:
                send_ints(socket, message, flags=flag)


    @staticmethod
    def create_help_message(address_base):
        return API_HELP_STRING_TEMPLATE.format(
            address_base=address_base,
            ping_address=SolverServer.ping_address(address_base),
            current_solution_address=SolverServer.current_solution_address(address_base),
            set_edge_labels_address=SolverServer.set_edge_labels_address(address_base),
            solution_update_request_address=SolverServer.solution_update_request_address(address_base),
            new_solution_address=SolverServer.new_solution_address(address_base),
            api_endpoint_address=SolverServer.api_endpoint_address(address_base))

    def __init__(
            self,
            context,
            directory,
            n5_container,
            paintera_dataset,
            next_solution_id = 0):
        super(SolverServer, self).__init__()

        if not SolverServer.is_paintera_data(n5_container, paintera_dataset):
            raise Exception('Dataset `{}\' is not paintera data in container `{}\''.format(paintera_dataset, n5_container))

        if not SolverServer.is_paintera_label_data(n5_container, paintera_dataset):
            raise Exception('Dataset `{}\' exists in container `{}\' but is not label data'.format(paintera_dataset, n5_container))

        edge_dataset = (paintera_dataset + '/' + SolverServer.default_edge_dataset()).strip('/')
        edge_feature_dataset = (paintera_dataset + '/' + SolverServer.default_edge_feature_dataset()).strip('/')

        os.makedirs(directory, exist_ok=True)

        self.pid = os.getpid()
        self.directory = directory
        self.lock_file = self.lock_directory()
        self.address_base = 'ipc://' + os.path.join(directory, 'server')
        self.logger = logging.getLogger('{}.{}'.format(self.__module__, type(self).__name__))
        self.logger.debug('Initializing workflow')
        self.save_lock = threading.RLock()
        self.workflow = Workflow(
            next_solution_id=next_solution_id, # TODO read from project file
            edge_n5_container=n5_container,
            edge_dataset=edge_dataset,
            edge_feature_dataset=edge_feature_dataset)
        self.logger.debug('Initialized workflow')

        def current_solution(_, socket):
            solution = self.workflow.get_latest_state()
            if solution is None or solution.solution is None:
                send_more_int(socket, _NO_SOLUTION_AVAILABLE)
                socket.send(b'')
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

        def api_socket_send(endpoint, socket):
            if len(endpoint) == 0 or endpoint == '' or endpoint == '/':
                socket.send(b'')
                return
            try:
                return_code = API_RESPONSE_OK
                message = '/' + endpoint.lstrip('/')
                # special case for ping
                if message == '/help' or message == 'help':
                    help_message = SolverServer.create_help_message(self.address_base)
                    self.logger.debug("Help message requested: %s", help_message)
                    messages = ((API_RESPONSE_DATA_STRING, help_message),)
                elif message == '/api/n5/all':
                    messages = ((API_RESPONSE_DATA_STRING, n5_container), (API_RESPONSE_DATA_STRING, paintera_dataset))
                elif message == '/api/n5/container':
                    messages = ((API_RESPONSE_DATA_STRING, n5_container),)
                elif message == '/api/n5/dataset':
                    messages = ((API_RESPONSE_DATA_STRING, paintera_dataset),)
                    self.logger.info('Collected dataset as message: %s', messages)
                elif message == '/api/save-ground-truth-labels':
                    exit_code = self.save_ground_truth()
                    self.logger.info('Saved ground truth: %d (0: success, 1: no data available)', exit_code)
                    messages = ((API_RESPONSE_DATA_INT, exit_code),)
                else:
                    return_code = API_RESPONSE_ENDPOINT_UNKNOWN
                    messages = ((API_RESPONSE_DATA_STRING, "Endpoint unknown"), (API_RESPONSE_DATA_STRING, endpoint))

            except Exception as e:
                return_code = API_RESPONSE_UNKNOWN_ERROR
                messages = tuple((API_RESPONSE_DATA_STRING, m) for m in (str(type(e)), str(e)))

            SolverServer.api_endpoint_respond(socket, return_code, *messages)



        self.ping_address                    = SolverServer.ping_address(self.address_base)
        self.current_solution_address        = SolverServer.current_solution_address(self.address_base)
        self.set_edge_labels_address         = SolverServer.set_edge_labels_address(self.address_base)
        self.solution_update_request_address = SolverServer.solution_update_request_address(self.address_base)
        self.new_solution_address            = SolverServer.new_solution_address(self.address_base)
        self.api_endpoint_address            = SolverServer.api_endpoint_address(self.address_base)

        api_socket                     = ReplySocket(self.api_endpoint_address, timeout=10, respond=api_socket_send)
        ping_socket                    = ReplySocket(self.ping_address, timeout=10)
        solution_notifier_socket       = PublishSocket(self.new_solution_address, timeout=10 / 1000, send=publish_new_solution) # queue timeout is specified in seconds
        solution_request_socket        = ReplySocket(self.current_solution_address, timeout=10, respond=current_solution)
        solution_update_request_socket = ReplySocket(self.solution_update_request_address, timeout=10, respond=update_request_received_confirmation)
        set_edge_labels_request_socket = ReplySocket(self.set_edge_labels_address, timeout=10, respond=set_edge_labels_send, receive=set_edge_labels_receive)

        self.workflow.add_solution_update_listener(lambda solution_id, exit_code, solution: solution_notifier_socket.queue.put((solution_id, exit_code)))


        self.context = context
        self.server  = Server(
            api_socket,
            ping_socket,
            solution_notifier_socket,
            solution_request_socket,
            set_edge_labels_request_socket,
            solution_update_request_socket)

        logging.info('Starting solver server at base address          %s', self.address_base)
        logging.info('Endpoint (send /help for more information)      %s', self.api_endpoint_address)
        logging.info('Ping server at                                  %s', self.ping_address)
        logging.info('Request current solution at                     %s', self.current_solution_address)
        logging.info('Submit edge labels at                           %s', self.set_edge_labels_address)
        logging.info('Request update of current solution at           %s', self.solution_update_request_address)
        logging.info('Subscribe to be notified about new solutions at %s', self.new_solution_address)

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

    def get_api_endpoint_address(self):
        return self.api_endpoint_address

    def shutdown(self):
        # TODO handle things like saving etc in here
        self.logger.debug('Shutting down server at base address %s', self.address_base)
        self.server.stop()
        self.workflow.stop()
        self.save_ground_truth()
        self.unlock_directory()

    def lock_directory(self):
        lock_file = os.path.join(self.directory, '.lock')
        # 'x': open for exclusive creation, failing if the file already exists
        # https://docs.python.org/3/library/functions.html#open
        try:
            with open(lock_file, 'x') as f:
                f.write(str(self.pid))
        except FileExistsError as e:
            # 17 is file exists
            assert e.errno == 17, 'Exception errno (%d) inconsistent with `file exists\' (17)' % e.errno
            raise Exception('File lock for server directory %s already exists at %s. '
                            'If you are certain that no other PIAS instance is running '
                            'on that directory delete %s and restart' % (self.directory, e.filename, e.filename))

        return lock_file

    def unlock_directory(self):
        if not self.lock_file:
            raise Exception('Directory %s not locked.' % self.directory)
        os.remove(self.lock_file)
        self.lock_file = None

    def save_ground_truth(self):
        state = self.workflow.get_latest_state()

        if state is None:
            return 1

        save_tmp_dir = os.path.join(self.directory, 'tmp')
        os.makedirs(save_tmp_dir, exist_ok=True)
        tmp_dir = tempfile.mkdtemp(prefix='ground-truth-', suffix='.n5', dir=save_tmp_dir)
        labels = state.labels
        uv_pairs = state.uv_pairs
        with z5py.File(tmp_dir, 'w') as f:
            f.create_dataset('labels', data=labels)
            f.create_dataset('edges', data=uv_pairs)

        ground_truth = os.path.join(self.directory, 'ground-truth.n5')
        with self.save_lock:
            while os.path.exists(ground_truth):
                try:
                    os.rename(ground_truth, ground_truth + '.' + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                except OSError as e:
                    # 39: Directory not empty
                    # 17: File exists
                    if e.errno not in (17, 39):
                        raise e

            os.rename(tmp_dir, ground_truth)

        return 0



def server_main(argv=None):
    import argparse
    from . import version
    parser = argparse.ArgumentParser()
    parser.add_argument('--container', required=True, help='N5 FS Container with group that contains edges as pairs of fragment labels and features')
    parser.add_argument('--paintera-dataset', required=True, help=f'Paintera dataset inside CONTAINER that also contains datasets `{_EDGE_DATASET}\' and `{_EDGE_FEATURE_DATASET}\'')
    parser.add_argument('--directory', required=False, help='Directory for ipc sockets and serialization of server state.', default='pias')
    parser.add_argument('--num-io-threads', required=False, type=int, default=1)
    parser.add_argument('--log-level', required=False, choices=log_levels, default='INFO')
    parser.add_argument('--version', action='version', version=f'{version}')

    args = parser.parse_args(args=argv)
    logging.basicConfig(level=logging.getLevelName(args.log_level))
    logger = logging.getLogger(__name__)

    context = zmq.Context(args.num_io_threads)
    try:
        server = SolverServer(
            context = context,
            n5_container=args.container,
            paintera_dataset=args.paintera_dataset,
            next_solution_id=0,
            directory=args.directory)

        def sigint_handler(signum, frame):
            logger.debug('Signal handler called with signal %s', signum)
            logger.info('Shutting down server at %s (%s)', server.directory, server.address_base)
            server.shutdown()
            context.destroy()

        signal.signal(signal.SIGINT, handler=sigint_handler)

    except Exception as e:
        logger.error('Unable to start server: %s', e)
        logger.debug('Exception info: %s', e, exc_info=True)
        context.destroy()

    # TODO add handler to shutdown server on ctrl-c

def client_cli_main(argv=None):
    import argparse
    import sys
    from . import version

    _logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    parser.add_argument('endpoint')
    parser.add_argument('--address', required=True, help='base address of server for which help is requested')
    parser.add_argument('--version', action='version', version=f'{version}')
    parser.add_argument('--log-level', required=False, choices=log_levels, default='INFO')

    args = parser.parse_args(args=argv)

    logging.basicConfig(level=logging.getLevelName(args.log_level))

    context = zmq.Context(1)
    socket = context.socket(zmq.REQ)
    socket.connect(args.address)
    socket.send_string(args.endpoint)
    response_code = recv_int(socket)

    if response_code != API_RESPONSE_OK:
        _logger.error('Received non-zero return code %d', response_code)
        sys.exit(response_code)

    for i in range(0, recv_int(socket)):
        message_type = recv_int(socket)
        if message_type == API_RESPONSE_DATA_STRING:
            data = socket.recv_string()
        elif message_type == API_RESPONSE_DATA_INT:
            data = recv_int(socket)
        elif message_type == API_RESPONSE_DATA_BYTES or API_RESPONSE_DATA_UNKNOWN:
            data = socket.recv()
        else:
            raise Exception('Do not understand message type %d', message_type)
        print(data)


