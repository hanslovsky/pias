from __future__ import absolute_import, division, print_function

import time

import logging
import threading
import zmq
try:
    import queue
except ImportError:
    import Queue as queue

_logger = logging.getLogger(__name__)


class StartStop(object):
    
    def __init__(self):
        super(StartStop, self).__init__()

    def start(self, context):
        pass

    def stop(self):
        pass

# def socket_send(socket, message, suffix = None, flags=0):
#     try:
#         send_more = len(message) > 1
#     except TypeError:
#         send_more = False
#         message = (message,)
#
#     send = socket.send if suffix is None else getattr(socket, 'send%s' % suffix, socket.send)
#
#     for m in message[:-1]:
#         send(m, flags=flags | zmq.SNDMORE)
#     send(message[-1], flags=flags & (~zmq.SNDMORE))



class ReplySocket(StartStop):


    def __init__(
            self,
            address,
            receive = lambda socket: socket.recv_string(),
            respond = lambda request, socket: socket.send_string(''),
            timeout = None,
            use_daemon = True,
            socket_send_suffix = None,
            socket_send_flags = 0):
        super(ReplySocket, self).__init__()

        self.logger = logging.getLogger('{}.{}'.format(self.__module__, type(self).__name__))
        self.logger.debug('Instantiating %s', type(self).__name__)

        self.address            = address
        self.socket             = None
        self.listening          = False
        self.receive            = receive
        self.respond            = respond
        self.thread             = None
        self.timeout            = timeout
        self.use_daemon         = use_daemon
        self.socket_send_flags  = socket_send_flags
        self.socket_send_suffix = socket_send_suffix

        self.logger.debug('Instantiated %s', self)

    def start(self, context):

        if self.listening:
            self.logger.debug('%s already bound')
            raise Exception("Socket already bound!")

        self.logger.debug('%s: starting with context %s', self, context)
        self.socket = context.socket(zmq.REP)
        self.socket.bind(self.address)
        if self.timeout is not None:
            self.socket.setsockopt(zmq.RCVTIMEO, self.timeout)

        socket         = self.socket
        respond        = self.respond
        self.listening = True

        this = self

        def task():
            while this.listening and this.socket:
                # this.logger.debug('%s: waiting to receive')
                try:
                    request = this.receive(socket)
                    this.logger.debug('%s: received `%s\'', this, request)
                except zmq.Again:
                    request = None
                except zmq.ContextTerminated:
                    break
                if request is not None:
                    respond(request, socket)
                    # this.logger.debug('%s: sending `%s\'', this, rep)
                    # socket_send(socket, rep, flags=self.socket_send_flags, suffix=self.socket_send_suffix)

        self.thread = threading.Thread(target=task, name='reply-on-%s' % self.address)
        self.thread.setDaemon(self.use_daemon)
        self.thread.start()

    def stop(self):
        self.listening = False

        if self.socket is not None:
            self.socket.unbind(self.address)
        self.socket = None

        if self.thread is not None:
            self.thread.join()
        self.thread = None

class PublishSocket(StartStop):


    def __init__(self, address, send = lambda socket, message: socket.send_string(message), timeout = 0., use_daemon=True, maxsize=0):
        super(PublishSocket, self).__init__()

        self.logger = logging.getLogger('{}.{}'.format(self.__module__, type(self).__name__))
        self.logger.debug('Instantiating %s', type(self).__name__)

        self.address    = address
        self.queue      = queue.Queue(maxsize=maxsize)
        self.timeout    = timeout
        self.socket     = None
        self.thread     = None
        self.use_daemon = use_daemon
        self.running    = False
        self.send       = send

        self.logger.debug('Instantiated %s', self)

    def start(self, context):

        if self.running:
            self.logger.debug('%s already bound')
            raise Exception("Socket already bound!")

        self.running = True

        self.logger.debug('%s: starting with context %s', self, context)
        self.socket = context.socket(zmq.PUB)
        self.socket.bind(self.address)

        socket = self.socket
        this   = self

        def task():
            while this.running and this.socket:
                # this.logger.debug('%s: waiting to receive')
                try:
                    item = this.queue.get(timeout=this.timeout)
                    this.logger.debug('%s: next item `%s\'', this, item)
                except queue.Empty:
                    item = None
                if item is not None:
                    this.logger.debug('%s: sending `%s\'', this, item)
                    this.send(socket, item)

        self.thread = threading.Thread(target=task, name='publish-on-%s' % self.address)
        self.thread.setDaemon(self.use_daemon)
        self.thread.start()

    def stop(self):
        self.running = False

        if self.socket is not None:
            self.socket.unbind(self.address)
        self.socket = None

        if self.thread is not None:
            self.thread.join()
        self.thread = None

    def __str__(self):
        return '%s[address=%s running=%s timeout=%s use_daemon=%s]' % (
            type(self).__name__,
            self.address,
            self.running,
            self.timeout,
            self.use_daemon)


class Server(object):

    def __init__(self, address_base):
        super(Server, self).__init__()

        self.logger = logging.getLogger('{}.{}'.format(self.__module__, type(self).__name__))
        self.logger.debug('Instantiating server!')

        self.address_base             = address_base
        self.socket                   = None
        self.ping_socket              = ReplySocket('%s-ping' % self.address_base, timeout=10)
        self.solution_notifier_socket = PublishSocket('%s-new-solution' % self.address_base, timeout=10 / 1000)
        self.lock = threading.RLock()

    def start(self, context):
        self.ping_socket.start(context)
        self.solution_notifier_socket.start(context)

    def stop(self):
        self.ping_socket.stop()
        self.solution_notifier_socket.stop()

    def __str__(self):
        return '%s[address=%s]' % (type(self).__name__, self.address_base)

if __name__ == "__main__":
    context = zmq.Context(1)
    logging.basicConfig(level=logging.DEBUG)

    address2 = 'inproc://lol2'
    subscriber = context.socket(zmq.SUB)
    subscriber.connect(address2)
    subscriber.setsockopt(zmq.SUBSCRIBE, b'')
    def recv():
        print("receiving string")
        edition = subscriber.recv_string()
        print("Got something!", edition)
    t = threading.Thread(target=recv)
    t.start()
    publisher = PublishSocket(address = address2)
    publisher.start(context)
    publisher.queue.put('123')
    t.join()
    publisher.stop()
