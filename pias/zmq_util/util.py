import numpy as np
import struct
import zmq

# big-endian:
# https://docs.python.org/2/library/struct.html#byte-order-size-and-alignment
_USE_BIG_ENDIAN  = True
_ENDIANNESS      = '>' if _USE_BIG_ENDIAN else '<'
_INTEGER_PATTERN = f'{_ENDIANNESS}i'
_UINT64_PATTERN  = f'{_ENDIANNESS}Q'
_EDGE_PATTERN    = 'QQi'


def _i32_pattern(n=1):
    i32s = 'i' * n
    return f'{_ENDIANNESS}{i32s}'

def _int_as_bytes(number):
    return struct.pack(_INTEGER_PATTERN, number)

def _ints_as_bytes(*numbers):
    return struct.pack(_i32_pattern(len(numbers)), *numbers)

def _bytes_as_int(b):
    return struct.unpack(_INTEGER_PATTERN, b)[0]

def _bytes_as_ints(b):
    # expect 4 bytes per int
    return struct.unpack(_i32_pattern(len(b) // 4), b)

def _edges_as_bytes(edges):
    pattern = f'{_ENDIANNESS}%s' % (_EDGE_PATTERN * len(edges))
    return struct.pack(pattern, *tuple(e for edge in edges for e in edge))

def _bytes_as_edges(b):
    # 8 + 8 + 4
    # label1, label2, 1 or 0
    # uint64,uint64,byte
    entry_size = 20
    a = len(b)
    assert len(b) % entry_size == 0, 'Message length is not integer multiple of entry size: 20 (8 + 8 + 4)'
    num_edges = len(b) // entry_size
    pattern = f'{_ENDIANNESS}%s' % (_EDGE_PATTERN * num_edges)
    e = struct.unpack(pattern, b)
    return tuple((e[i+0], e[i+1], e[i+2]) for i in range(0, len(e), 3))

def _ndarray_as_bytes(ndarray):
    # java always big endian
    # https://stackoverflow.com/questions/981549/javas-virtual-machines-endianness
    return (ndarray.byteswap() if _USE_BIG_ENDIAN else ndarray).tobytes()

def _bytes_as_ndarray(buffer, dtype, count=-1, offset=0):
    # java always big endian
    # https://stackoverflow.com/questions/981549/javas-virtual-machines-endianness
    ndarray = np.frombuffer(buffer, dtype=dtype, count=-1, offset=0)
    return ndarray.byteswap() if _USE_BIG_ENDIAN else ndarray

def send_int(socket, number, flags=0, copy=True, track=False, routing_id=None, group=None):
    socket.send(_int_as_bytes(number), flags=flags, copy=copy, track=track, routing_id=routing_id, group=group)

def send_ints(socket, *numbers, flags=0, copy=True, track=False, routing_id=None, group=None):
    socket.send(_ints_as_bytes(*numbers), flags=flags, copy=copy, track=track, routing_id=routing_id, group=group)

def send_more_int(socket, number, flags=0, copy=True, track=False, routing_id=None, group=None):
    socket.send(_int_as_bytes(number), flags=flags | zmq.SNDMORE, copy=copy, track=track, routing_id=routing_id, group=group)

def recv_int(socket, flags=0, copy=True, track=False):
    return _bytes_as_int(socket.recv(flags=flags, copy=copy, track=track))

def recv_ints(socket, flags=0, copy=True, track=False):
    return _bytes_as_ints(socket.recv(flags=flags, copy=copy, track=track))

def send_ints_multipart(socket, *numbers, flags=0, copy=True, track=False, **kwargs):
    socket.send_multipart(msg_parts=tuple(_int_as_bytes(n) for n in numbers), flags=flags, copy=copy, track=track, **kwargs)

def recv_ints_multipart(socket, flags=0, copy=True, track=False):
    return tuple(_bytes_as_int(b) for b in socket.recv_multipart(flags=flags, copy=copy, track=track))