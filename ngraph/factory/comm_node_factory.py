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
from .comm_nodes import GpuQueueSendOp, GpuQueueRecvOp, NumpyQueueSendOp, \
    NumpyQueueRecvOp, NumpyQueueGatherSendOp, NumpyQueueGatherRecvOp, \
    NumpyQueueScatterSendOp, NumpyQueueScatterRecvOp
from ngraph.op_graph.op_graph import BroadcastOp
from collections import defaultdict


class CommNodePair(object):
    """
    Represents a communication pair (sender, receiver).

    Arguments:
        from_node: The source node.
        to_node: The destination node.
        node_type: The type of node (direct/scatter/gather).
    """

    def __init__(self, from_node, to_node, node_type):

        def get_node_factory(node):
            if node.metadata['device'] == 'gpu':
                return GpuCommNodeFactory()
            elif node.metadata['device'] == 'numpy':
                return NumpyCommNodeFactory()
            else:
                assert False

        def get_location(from_node, to_node):
            (send_host, send_idx) = from_node.metadata['host_transformer']
            (recv_host, recv_idx) = to_node.metadata['host_transformer']

            if send_host == recv_host:
                return 'local'
            else:
                return 'remote'

        def get_comm_type(comm_options_send, comm_options_recv):
            for send_comm_type in comm_options_send:
                for recv_comm_type in comm_options_recv:
                    if send_comm_type == recv_comm_type:
                        return send_comm_type
            assert False, "Not compatible!!!"

        self.send_node = None
        self.recv_node = None

        send_node_factory = get_node_factory(from_node)
        recv_node_factory = get_node_factory(to_node)

        location = get_location(from_node, to_node)

        comm_options_send = send_node_factory.send_recv_types(location)
        comm_options_recv = recv_node_factory.send_recv_types(location)

        comm_type = get_comm_type(comm_options_send, comm_options_recv)

        if node_type == 'scatter':
            self.send_node = send_node_factory.build(
                node_type='scatter_send',
                comm_type=comm_type,
                from_node=from_node,
                to_node=to_node)
            self.recv_node = recv_node_factory.build(
                node_type='scatter_recv',
                comm_type=comm_type,
                from_node=from_node,
                to_node=to_node,
                send_node=self.send_node)
        elif node_type == 'gather':
            self.send_node = send_node_factory.build(
                node_type='gather_send',
                comm_type=comm_type,
                from_node=from_node,
                to_node=to_node)
            self.recv_node = recv_node_factory.build(
                node_type='gather_recv',
                comm_type=comm_type,
                from_node=from_node,
                to_node=to_node,
                send_node=self.send_node)
        elif node_type == 'direct':
            self.send_node = send_node_factory.build(
                node_type='send',
                comm_type=comm_type,
                from_node=from_node)
            self.recv_node = recv_node_factory.build(
                node_type='recv',
                comm_type=comm_type,
                to_node=to_node,
                send_node=self.send_node)

    def get_nodes(self):
        return self.send_node, self.recv_node


class CommNodeFactory(object):
    """
    Represents a communication node factory.

    Arguments:
        None
    """

    def __init__(self):
        pass

    def send_recv_types(self):
        pass

    def build(self, node_type, comm_type, from_node=None, to_node=None, send_node=None):
        pass


class GpuCommNodeFactory(CommNodeFactory):
    """
    Represents a GPU communication node factory.

    Arguments:
        None
    """

    def send_recv_types(self, location):
        types = [
            ('remote', 'mpi'),
            ('local', 'queue'),
            ('local', 'cuda')
        ]

        send_recv_types = defaultdict(list)
        for loc, comm_type in types:
            send_recv_types[loc].append(comm_type)

        return send_recv_types[location]

    def build(self, node_type, comm_type, from_node=None, to_node=None, send_node=None):
        if node_type == 'send':
            if comm_type == 'queue':
                return GpuQueueSendOp(
                    from_node=from_node)
        elif node_type == 'recv':
            if comm_type == 'queue':
                return GpuQueueRecvOp(
                    to_node=to_node,
                    send_node=send_node)
        else:
            assert False, "Not supported!!!"


class NumpyCommNodeFactory(CommNodeFactory):
    """
    Represents a NumPy communication node factory.

    Arguments:
        None
    """

    def send_recv_types(self, location):
        types = [
            ('remote', 'mpi'),
            ('local', 'queue'),
        ]

        send_recv_types = defaultdict(list)
        for loc, comm_type in types:
            send_recv_types[loc].append(comm_type)

        return send_recv_types[location]

    def build(self, node_type, comm_type, from_node=None, to_node=None, send_node=None):
        if node_type == 'send':
            if comm_type == 'queue':
                return NumpyQueueSendOp(
                    from_node=from_node)
        elif node_type == 'recv':
            if comm_type == 'queue':
                return NumpyQueueRecvOp(
                    to_node=to_node,
                    send_node=send_node)
        elif node_type == 'scatter_send':
            if comm_type == 'queue':
                return NumpyQueueScatterSendOp(
                    from_node=from_node,
                    to_node=to_node)
        elif node_type == 'scatter_recv':
            if comm_type == 'queue':
                return NumpyQueueScatterRecvOp(
                    to_node=to_node,
                    send_node=send_node)
        elif node_type == 'gather_send':
            if comm_type == 'queue':
                return NumpyQueueGatherSendOp(
                    from_node=from_node)
        elif node_type == 'gather_recv':
            if comm_type == 'queue':
                return NumpyQueueGatherRecvOp(
                    from_node=from_node,
                    to_node=to_node,
                    send_node=send_node)
        else:
            assert False, "Not supported!!!"


def get_node_type(from_node, to_node):
    if isinstance(to_node.metadata['device_id'], (list, tuple)):
        if isinstance(from_node, BroadcastOp):
            if from_node.args[0].is_constant:
                return None
        elif not from_node.is_constant:
            from_node.metadata['marker'] = 'scatter'
            return 'scatter'
        else:
            return None
    elif isinstance(from_node.metadata['device_id'], (list, tuple)):
        return 'gather'
    elif from_node.metadata['device_id'] != to_node.metadata['device_id']:
        return 'direct'
    elif from_node.metadata['device'] != to_node.metadata['device']:
        return 'direct'
    else:
        return None
