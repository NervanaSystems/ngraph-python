# ----------------------------------------------------------------------------
# Copyright 2017 Nervana Systems Inc.
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
from functools import reduce


class HetrLocals(object):
    def __init__(self, send_nodes, recv_nodes,
                 scatter_send_nodes, scatter_recv_nodes,
                 gather_send_nodes, gather_recv_nodes,
                 allreduce_nodes, broadcast_send_nodes,
                 broadcast_recv_nodes, **kwargs):
        super(HetrLocals, self).__init__(**kwargs)
        self.send_nodes = send_nodes
        self.recv_nodes = recv_nodes
        self.scatter_send_nodes = scatter_send_nodes
        self.scatter_recv_nodes = scatter_recv_nodes
        self.gather_send_nodes = gather_send_nodes
        self.gather_recv_nodes = gather_recv_nodes
        self.allreduce_nodes = allreduce_nodes
        self.broadcast_send_nodes = broadcast_send_nodes
        self.broadcast_recv_nodes = broadcast_recv_nodes

    def queue_send(self, send_id, x_nparr):
        send_op = self.send_nodes[send_id]
        q = send_op.queue

        # TODO
        # below converts DeviceTensor to numpy array
        # should we instead serialize DeviceTensor?
        q.put(x_nparr)

    def recv_from_queue_send(self, recv_id, out):
        recv_op = self.recv_nodes[recv_id]
        q = recv_op.queue
        out[...] = q.get()
        return out

    def queue_gather_send(self, gather_send_id, x_nparr):
        gather_send_op = self.gather_send_nodes[gather_send_id]
        q = gather_send_op.shared_queues[gather_send_op.idx]
        # TODO
        # below converts DeviceTensor to numpy array
        # should we instead serialize DeviceTensor?
        q.put(x_nparr)

    def gather_recv_from_queue_gather_send(self, gather_recv_id, out):
        gather_recv_op = self.gather_recv_nodes[gather_recv_id]
        for i in range(len(gather_recv_op.from_id)):
            q = gather_recv_op.shared_queues[i]
            x = q.get()
            out[gather_recv_op.slices[i]] = x
        return out

    def queue_scatter_send(self, scatter_send_id, x_nparr):
        scatter_send_op = self.scatter_send_nodes[scatter_send_id]
        # TODO
        # below converts DeviceTensor to numpy array
        # should we instead serialize DeviceTensor?
        for i in range(len(scatter_send_op.to_id)):
            q = scatter_send_op.shared_queues[i]
            q.put(x_nparr[scatter_send_op.slices[i]])

    def scatter_recv_from_queue_scatter_send(self, scatter_recv_id, out):
        scatter_recv_op = self.scatter_recv_nodes[scatter_recv_id]
        q = scatter_recv_op.shared_queues[scatter_recv_op.idx]
        out[...] = q.get()
        return out

    def queue_allreduce(self, allreduce_id, x_nparr):
        allreduce_op = self.allreduce_nodes[allreduce_id]
        recv_buf = list()

        # Send to all devices
        for i, q in enumerate(allreduce_op.shared_queues):
            if i != allreduce_op.idx:
                q.put(x_nparr)

        # Receive from all devices
        recv_buf.append(x_nparr)
        q = allreduce_op.shared_queues[allreduce_op.idx]
        for i in range(len(allreduce_op.shared_queues) - 1):
            recv_buf.append(q.get())

        # Apply reduce function
        if allreduce_op.reduce_func is 'sum':
            result = reduce(lambda x, y: x + y, recv_buf)
        elif allreduce_op.reduce_func is 'mean':
            result = reduce(lambda x, y: x + y, recv_buf) / len(recv_buf)

        return result

    def queue_broadcast_send(self, broadcast_send_id, x_nparr):
        broadcast_send_op = self.broadcast_send_nodes[broadcast_send_id]
        for i in range(len(broadcast_send_op.to_id)):
            q = broadcast_send_op.shared_queues[i]
            q.put(x_nparr)

    def broadcast_recv_from_queue_broadcast_send(self, broadcast_recv_id, out):
        broadcast_recv_op = self.broadcast_recv_nodes[broadcast_recv_id]
        q = broadcast_recv_op.shared_queues[broadcast_recv_op.idx]
        out[...] = q.get()
        return out
