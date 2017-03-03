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
from ngraph.op_graph.op_graph import TensorOp
import multiprocessing


class CommunicationOp(TensorOp):

    def __init__(self):
        super(CommunicationOp, self).__init__()

    @property
    def is_communication_op(self):
        return True


class SendOp(CommunicationOp):

    def __init__(self, from_node):
        super(SendOp, self).__init__()
        self._Op__args = tuple([from_node])
        self.axes = from_node.axes
        self.dtype = from_node.dtype
        self.metadata['device'] = from_node.metadata['device']
        self.metadata['device_id'] = from_node.metadata['device_id']
        self.metadata['host_transformer'] = from_node.metadata['host_transformer']


class ReceiverOp(CommunicationOp):

    def __init__(self, send_node):
        super(ReceiverOp, self).__init__()
        self._send_node = send_node

    def send_node(self):
        return self._send_node


class RecvOp(ReceiverOp):

    def __init__(self, to_node, send_node):
        super(RecvOp, self).__init__(send_node)
        self._TensorOp__args = ()
        self._TensorOp__axes = to_node.axes
        self.dtype = to_node.dtype
        self.metadata['device'] = to_node.metadata['device']
        self.metadata['device_id'] = to_node.metadata['device_id']
        self.metadata['host_transformer'] = to_node.metadata['host_transformer']


class ScatterSendOp(SendOp):

    def __init__(self, from_node, to_node):
        super(ScatterSendOp, self).__init__(from_node)
        parallel_axis = to_node.metadata['parallel']
        self.to_id = to_node.metadata['device_id']
        self.slices = list()
        for i in range(len(self.to_id)):
            slices = list()
            for a in self.axes:
                s = slice(None)
                if parallel_axis == a:
                    remainder = a.length % len(self.to_id)
                    new_length = a.length // len(self.to_id)
                    start = i * new_length
                    stop = (i + 1) * new_length
                    step = 1
                    if remainder > 0:
                        if i == (len(self.to_id) - 1):
                            stop += remainder
                    s = slice(start, stop, step)
                slices.append(s)
            self.slices.append(slices)


class ScatterRecvOp(RecvOp):

    def __init__(self, to_node, send_node):
        super(ScatterRecvOp, self).__init__(to_node, send_node)


class GatherSendOp(SendOp):

    def __init__(self, from_node):
        super(GatherSendOp, self).__init__(from_node)


class GatherRecvOp(RecvOp):

    def __init__(self, from_node, to_node, send_node):
        super(GatherRecvOp, self).__init__(to_node, send_node)
        self.metadata['marker'] = 'gather'
        self.metadata['parallel'] = from_node.metadata['parallel']
        self.from_id = from_node.metadata['device_id']
        self.slices = list()
        for i in range(len(self.from_id)):
            slices = list()
            for a in self.axes:
                s = slice(None)
                if self.metadata['parallel'] == a:
                    remainder = a.length % len(self.from_id)
                    new_length = a.length // len(self.from_id)
                    start = i * new_length
                    stop = (i + 1) * new_length
                    step = 1
                    if remainder > 0:
                        if i == (len(self.from_id) - 1):
                            stop += remainder
                    s = slice(start, stop, step)
                slices.append(s)
            self.slices.append(slices)


class GpuQueueSendOp(SendOp):

    def __init__(self, from_node):
        super(GpuQueueSendOp, self).__init__(from_node)
        self.queue = multiprocessing.Queue()


class GpuQueueRecvOp(RecvOp):

    def __init__(self, to_node, send_node):
        super(GpuQueueRecvOp, self).__init__(to_node, send_node)
        self.queue = send_node.queue


class NumpyQueueSendOp(SendOp):

    def __init__(self, from_node):
        super(NumpyQueueSendOp, self).__init__(from_node)
        self.queue = multiprocessing.Queue()


class NumpyQueueRecvOp(RecvOp):

    def __init__(self, to_node, send_node):
        super(NumpyQueueRecvOp, self).__init__(to_node, send_node)
        self.queue = send_node.queue


class NumpyQueueScatterSendOp(ScatterSendOp):

    def __init__(self, from_node, to_node):
        super(NumpyQueueScatterSendOp, self).__init__(from_node, to_node)
        self.shared_queues = list()
        for i in range(len(to_node.metadata['device_id'])):
            self.shared_queues.append(multiprocessing.Queue())
        self.comm_type = 'queue'


class NumpyQueueScatterRecvOp(ScatterRecvOp):

    def __init__(self, to_node, send_node, device_idx=None):
        super(NumpyQueueScatterRecvOp, self).__init__(to_node, send_node)
        if device_idx:
            self.idx = device_idx
        else:
            self.idx = 0
        self.shared_queues = send_node.shared_queues


class NumpyQueueGatherSendOp(GatherSendOp):

    def __init__(self, from_node, clone_node=None, device_idx=None):
        super(NumpyQueueGatherSendOp, self).__init__(from_node)
        self.shared_queues = list()
        if clone_node:
            self.idx = device_idx
            self.shared_queues = clone_node.shared_queues
        else:
            self.idx = 0
            for i in range(len(from_node.metadata['device_id'])):
                self.shared_queues.append(multiprocessing.Queue())


class NumpyQueueGatherRecvOp(GatherRecvOp):

    def __init__(self, from_node, to_node, send_node):
        super(NumpyQueueGatherRecvOp, self).__init__(from_node, to_node, send_node)
        self.shared_queues = send_node.shared_queues
