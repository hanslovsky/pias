# Paintera Interactive Agglomeration Server

`pias` provides a server for interactive updates of a [`nifty`](https://github.com/DerThorsten/nifty)-based agglomeration model for fragments in EM segmentation (but could probably be used in other scenarios, as well). Clients provide positive (two fragments are in the same segment) and negative (two fragments should not be in the same segment) examples for edges (fragment-pairs). Random forest classifiers are trained on these examples to infer weights for all edges of neighboring fragments in a *multi-cut* problem (constraints for the enforcement of positive/negative examples provided by the client is modeled via appropriate weights).

## Installation

``` shell
pip install git+https://github.com/saalfeldlab/pias
```

## Server Details


[`pyzmq`](https://github.com/zeromq/pyzmq) is used for communication between server and client. Data should be sent as big endian. The server can be started with the `pias` command that is installed with the python package. See `pias --help` for usage details. Once started, the server will start multiple sockets, addressed by extensions of the `address_base` parameter:

  - `${address_base}-ping`             - ping the server at this address to see if it is alive (`REQ/REP`)
  - `${address_base}-current-solution` - request current solution (`REQ/REP`)
  - `${address_base}-set-edge-labels`  - set labels for edges: (multiples of) `(e1, e2, label)` (`REQ/REP`)
  - `${address_base}-update-solution`  - request update of current solution (`REQ/REP`)
  - `${address_base}-new-solution`     - be notified about updates of the current solution (`PUB/SUB`)

**NOTE**: This scheme probably works (reliably) with `ipc://` zmq-addresses.


### Socket Details

**TBD**: *What kind of input/output does each socket expect/provide?*
