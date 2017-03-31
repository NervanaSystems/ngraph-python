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


class HetrComputation(object):
    def __init__(self, send_nodes, recv_nodes,
                 scatter_send_nodes, scatter_recv_nodes,
                 gather_send_nodes, gather_recv_nodes,
                 **kwargs):
        super(HetrComputation, self).__init__(**kwargs)
        self.send_nodes = send_nodes
        self.recv_nodes = recv_nodes
        self.scatter_send_nodes = scatter_send_nodes
        self.scatter_recv_nodes = scatter_recv_nodes
        self.gather_send_nodes = gather_send_nodes
        self.gather_recv_nodes = gather_recv_nodes

    def queue_send(self, send_id):
        send_op = self.send_nodes[send_id]
        q = send_op.queue

        # TODO
        # below converts DeviceTensor to numpy array
        # should we instead serialize DeviceTensor?
        x_devicetensor = send_op.args[0].value
        x_nparr = x_devicetensor.get(None)
        q.put(x_nparr)

    def recv_from_queue_send(self, recv_id):
        recv_op = self.recv_nodes[recv_id]
        q = recv_op.queue
        x = q.get()
        return x

    def queue_gather_send(self, gather_send_id):
        gather_send_op = self.gather_send_nodes[gather_send_id]
        q = gather_send_op.shared_queues[gather_send_op.idx]
        # TODO
        # below converts DeviceTensor to numpy array
        # should we instead serialize DeviceTensor?
        x_devicetensor = gather_send_op.args[0].value
        x_nparr = x_devicetensor.get(None)
        q.put(x_nparr)

    def gather_recv_from_queue_gather_send(self, gather_recv_id):
        gather_recv_op = self.gather_recv_nodes[gather_recv_id]
        x_devicetensor = gather_recv_op.value
        x_nparr = x_devicetensor.get(None)
        for i in range(len(gather_recv_op.from_id)):
            q = gather_recv_op.shared_queues[i]
            x = q.get()
            x_nparr[gather_recv_op.slices[i]] = x
        return x_nparr

    def queue_scatter_send(self, scatter_send_id):
        scatter_send_op = self.scatter_send_nodes[scatter_send_id]
        # TODO
        # below converts DeviceTensor to numpy array
        # should we instead serialize DeviceTensor?
        x_devicetensor = scatter_send_op.args[0].value
        x_nparr = x_devicetensor.get(None)
        for i in range(len(scatter_send_op.to_id)):
            q = scatter_send_op.shared_queues[i]
            q.put(x_nparr[scatter_send_op.slices[i]])

    def scatter_recv_from_queue_scatter_send(self, scatter_recv_id):
        scatter_recv_op = self.scatter_recv_nodes[scatter_recv_id]
        q = scatter_recv_op.shared_queues[scatter_recv_op.idx]
        x = q.get()
        return x
