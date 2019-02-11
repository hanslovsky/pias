from __future__ import print_function

import time

import threading
import zmq

import unittest

from pias import ReplySocket, Server, PublishSocket


class TestReqSocket(unittest.TestCase):

    def test(self):
        address = 'inproc://test-req-socket'
        context = zmq.Context(io_threads=1)

        try:
            req = context.socket(zmq.REQ)
            req.connect(address)

            rep = ReplySocket(address, timeout=1, use_daemon=False, respond=lambda req, socket: socket.send_string('response'), socket_send_suffix='_string')
            rep.start(context)

            req.send_string('')
            response = req.recv_string()
            self.assertEqual('response', response)

            rep.stop()

            req2 = context.socket(zmq.REQ)
            req2.setsockopt(zmq.RCVTIMEO, 1)
            req2.connect(address)
            req2.send_string('')
            self.assertRaises(zmq.Again, req2.recv_string)

        finally:
            context.destroy()

class TestPubSocket(unittest.TestCase):

    def test(self):
        address = 'inproc://test-req-socket'
        context = zmq.Context(io_threads=1)

        try:
            subscriber = context.socket(zmq.SUB)
            subscriber.connect(address)
            subscriber.setsockopt(zmq.SUBSCRIBE, b'')

            responses = []
            def recv():
                responses.append(subscriber.recv_string())

            subscription_thread = threading.Thread(target=recv)
            subscription_thread.start()
            time.sleep(0.05)

            publisher = PublishSocket(address, timeout=0.1)
            publisher.start(context)
            publisher.queue.put('123')
            publisher.stop()
            subscription_thread.join(timeout=0.01)

            self.assertEqual(1, len(responses))
            self.assertEqual('123', responses[0])


        finally:
            context.destroy()


class TestServerBasic(unittest.TestCase):

    def test(self):
        address = 'inproc://test-server-socket'
        context = zmq.Context(io_threads=1)

        try:
            req = context.socket(zmq.REQ)
            req.connect('%s-ping' % address)

            server = Server(address_base=address)
            server.start(context)

            req.send_string('')
            response = req.recv_string()
            self.assertEqual('', response)

            server.stop()

            req2 = context.socket(zmq.REQ)
            req2.setsockopt(zmq.RCVTIMEO, 1)
            req2.connect('%s-ping' % address)
            req2.send_string('')
            self.assertRaises(zmq.Again, req2.recv_string)

        finally:
            context.destroy()