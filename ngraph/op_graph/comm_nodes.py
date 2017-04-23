# ----------------------------------------------------------------------------
# Copyright 2016 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
from __future__ import division
from ngraph.op_graph.op_graph import TensorOp, make_axes, make_axis, compute_reduction_axes
from orderedset import OrderedSet
import multiprocessing


def calculate_gather_axes(axes, gather_axis, num_devices):
    new_axes = [make_axis(a.length * num_devices, a.name)
                if gather_axis == a else a for a in axes]
    new_axes = make_axes(new_axes)
    return new_axes


def calculate_scatter_axes(axes, scatter_axis, num_devices):
    new_axes = list()
    for a in axes:
        if scatter_axis == a:
            assert a.length % num_devices == 0, '{} can not be equally paralleled by {}'\
                .format(scatter_axis, num_devices)

            new_length = a.length // num_devices
            new_axis = make_axis(new_length, a.name)
            new_axes.append(new_axis)
        else:
            new_axes.append(a)
    new_axes = make_axes(new_axes)
    return new_axes


def get_slices(axes, parallel_axis, num_devices):
    new_slices = list()
    for i in range(num_devices):
        slices = list()
        for a in axes:
            s = slice(None)
            if parallel_axis == a:
                remainder = a.length % num_devices
                new_length = a.length // num_devices
                start = i * new_length
                stop = (i + 1) * new_length
                step = 1
                if remainder > 0:
                    if i == (num_devices - 1):
                        stop += remainder
                s = slice(start, stop, step)
            slices.append(s)
        new_slices.append(slices)
    return new_slices


class ResultOp(TensorOp):
    """
    special op for Hetr distributed case, not supported by other transformers
    note: possible deprecation in future (issue #1115)
    """

    def __init__(self, device_id, args, **kwargs):

        super(ResultOp, self).__init__(args=args)
        self.metadata['device_id'] = device_id
        self.axes = args[0].axes
        self.dtype = args[0].dtype


class CommunicationOp(TensorOp):
    """
    Represents a communication op.

    Arguments:
        None
    """

    def __init__(self, node, args=None, axes=None, dtype=None):
        super(CommunicationOp, self).__init__(args=args, axes=axes, dtype=dtype)
        self.metadata['device'] = node.metadata['device']
        self.metadata['device_id'] = node.metadata['device_id']
        self.metadata['transformer'] = node.metadata['transformer']
        self.metadata['host_transformer'] = node.metadata['host_transformer']

    @property
    def is_communication_op(self):
        return True


class SendOp(CommunicationOp):
    """
    Represents a send op. Sets args, axes, dtype, and metadata.

    Arguments:
        from_node: The source node.
    """

    def __init__(self, from_node):
        super(SendOp, self).__init__(
            node=from_node,
            args=tuple([from_node]),
            axes=from_node.axes,
            dtype=from_node.dtype)


class RecvOp(CommunicationOp):
    """
    Represents a recv op. Sets args, axes, dtype, and metadata.

    Arguments:
        to_node: The destination node.
        send_node: The send node associated with this recv node.
    """

    def __init__(self, to_node, send_node, fragment_axis=None, fragments=None):
        super(RecvOp, self).__init__(
            node=to_node,
            args=(),
            axes=self.calculate_recv_axes(send_node.axes, fragment_axis, fragments),
            dtype=send_node.dtype)
        self._send_node = send_node

    @classmethod
    def calculate_recv_axes(cls, send_axes, fragment_axis, fragments):
        return send_axes

    def send_node(self):
        return self._send_node


class ScatterSendOp(SendOp):
    """
    Represents a scatter send op. Sets destination device ids and slices.

    Arguments:
        from_node: The source node.
        to_node: The destination node.
    """

    def __init__(self, from_node, to_node):
        super(ScatterSendOp, self).__init__(from_node)
        self.to_id = to_node.metadata['device_id']
        self._slices = get_slices(self.axes,
                                  to_node.metadata['parallel'],
                                  len(self.to_id))

    @property
    def slices(self):
        return self._slices


class ScatterRecvOp(RecvOp):
    """
    Represents a scatter recv op.

    Arguments:
        to_node: The destination node.
        send_node: The scatter send node associated with this node.
    """

    def __init__(self, to_node, send_node):
        super(ScatterRecvOp, self).__init__(to_node, send_node,
                                            fragment_axis=to_node.metadata['parallel'],
                                            fragments=len(to_node.metadata['device_id']))


class GatherSendOp(SendOp):
    """
    Represents a gather send op.

    Arguments:
        from_node: The source node.
    """

    def __init__(self, from_node):
        super(GatherSendOp, self).__init__(from_node)


class GatherRecvOp(RecvOp):
    """
    Represents a gather recv op. Sets metadata, source device ids and
    slices.

    Arguments:
        from_node: The source node.
        to_node: The destination node.
        send_node: The gather send node associated with this node.
    """

    def __init__(self, from_node, to_node, send_node):
        super(GatherRecvOp, self).__init__(to_node, send_node,
                                           fragment_axis=from_node.metadata['parallel'],
                                           fragments=len(from_node.metadata['device_id']))
        self.metadata['marker'] = 'gather'
        self.metadata['parallel'] = from_node.metadata['parallel']
        self.from_id = from_node.metadata['device_id']
        # use _slices to avoid serialization
        self._slices = get_slices(self.axes,
                                  self.metadata['parallel'],
                                  len(self.from_id))

    @property
    def slices(self):
        return self._slices

    @property
    def send_nodes(self):
        """
        :return: iterable of send nodes
        """
        from ngraph.util.hetr_utils import get_iterable
        return OrderedSet(i for i in get_iterable(self._send_node))

    @send_nodes.setter
    def send_nodes(self, new_send_nodes):
        self._send_node = new_send_nodes

    def send_node(self):
        # make it work for general traversal in functions (e.g. find_recv())
        return self.send_nodes


class GPUQueueSendOp(SendOp):

    def __init__(self, from_node):
        super(GPUQueueSendOp, self).__init__(from_node)
        self._queue = multiprocessing.Queue()

    @property
    def queue(self):
        return self._queue


class GPUQueueRecvOp(RecvOp):

    def __init__(self, to_node, send_node):
        super(GPUQueueRecvOp, self).__init__(to_node, send_node)
        self._queue = send_node.queue

    @property
    def queue(self):
        return self._queue


class CPUQueueSendOp(SendOp):

    def __init__(self, from_node):
        super(CPUQueueSendOp, self).__init__(from_node)
        self._queue = multiprocessing.Queue()

    @property
    def queue(self):
        return self._queue


class CPUQueueRecvOp(RecvOp):

    def __init__(self, to_node, send_node):
        super(CPUQueueRecvOp, self).__init__(to_node, send_node)
        self._queue = send_node.queue

    @property
    def queue(self):
        return self._queue


class CPUQueueScatterSendOp(ScatterSendOp):

    def __init__(self, from_node, to_node):
        super(CPUQueueScatterSendOp, self).__init__(from_node, to_node)
        self._shared_queues = [multiprocessing.Queue() for i in to_node.metadata['device_id']]
        self.comm_type = 'queue'

    @property
    def shared_queues(self):
        return self._shared_queues


class CPUQueueScatterRecvOp(ScatterRecvOp):

    def __init__(self, to_node, send_node):
        super(CPUQueueScatterRecvOp, self).__init__(to_node, send_node)
        self.idx = 0
        self._shared_queues = send_node.shared_queues

    @property
    def shared_queues(self):
        return self._shared_queues


class CPUQueueGatherSendOp(GatherSendOp):

    def __init__(self, from_node):
        super(CPUQueueGatherSendOp, self).__init__(from_node)
        self.idx = 0
        self._shared_queues = [multiprocessing.Queue() for i in from_node.metadata['device_id']]

    @property
    def shared_queues(self):
        return self._shared_queues


class CPUQueueGatherRecvOp(GatherRecvOp):

    def __init__(self, from_node, to_node, send_node):
        super(CPUQueueGatherRecvOp, self).__init__(from_node, to_node, send_node)
        self._shared_queues = send_node.shared_queues

    @property
    def shared_queues(self):
        return self._shared_queues


# TODO : WIP. This will be updated once we define the logic in issue #1378
class AllReduceOp(CommunicationOp):

    def __init__(self, x, func, reduction_axes=None, out_axes=None, dtype=None, **kwargs):
        reduction_axes, out_axes = compute_reduction_axes(x, reduction_axes, out_axes)
        self.func = func
        self.reduction_axes = reduction_axes
        super(AllReduceOp, self).__init__(node=x, axes=out_axes, dtype=dtype, **kwargs)


class MeanAllReduceOp(AllReduceOp):

    def __init__(self, x, reduction_axes=None, out_axes=None, dtype=None, **kwargs):
        super(MeanAllReduceOp, self).__init__(x=x,
                                              func='mean',
                                              reduction_axes=reduction_axes,
                                              out_axes=out_axes,
                                              dtype=dtype,
                                              **kwargs)
