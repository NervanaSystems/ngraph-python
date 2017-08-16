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
from ngraph.op_graph.comm_nodes import SendOp, ScatterSendOp, GatherSendOp, RecvOp, \
    ScatterRecvOp, GatherRecvOp
import ngraph as ng


def create_send_recv_graph():
    ax_a = ng.make_axis(length=10, name='A')
    ax_b = ng.make_axis(length=15, name='B')
    axes = ng.make_axes([ax_a, ax_b])

    with ng.metadata(parallel=ax_a, device=None, device_id=None,
                     transformer=None, host_transformer=None):
        from_node = ng.placeholder(axes)
        to_node = ng.placeholder(axes)
    send_x = SendOp(from_node=from_node)
    recv_x = RecvOp(to_node=to_node, send_node=send_x)

    with ng.metadata(parallel=ax_a, device=None, device_id=None,
                     transformer=None, host_transformer=None):
        x_plus_one = recv_x + 1

    send_x_plus_one = SendOp(from_node=x_plus_one)
    recv_x_plus_one = RecvOp(to_node=to_node, send_node=send_x_plus_one)

    with ng.metadata(device=None, device_id=None, transformer=None, host_transformer=None):
        z = recv_x_plus_one + 2
    return z, recv_x, recv_x_plus_one, send_x, x_plus_one, from_node, send_x_plus_one


def create_scatter_gather_graph():
    ax_a = ng.make_axis(length=10, name='A')
    ax_b = ng.make_axis(length=20, name='B')
    axes = ng.make_axes([ax_a, ax_b])

    with ng.metadata(parallel=ax_b, device=(0, 1), device_id=(0, 1),
                     transformer=None, host_transformer=None):
        from_node = ng.placeholder(axes)
        to_node = ng.placeholder(axes)
    scatter_send_x = ScatterSendOp(from_node=from_node, to_node=to_node)
    scatter_recv_a = ScatterRecvOp(to_node=to_node, send_node=scatter_send_x)
    scatter_recv_b = ScatterRecvOp(to_node=to_node, send_node=scatter_send_x)
    gather_send_a = GatherSendOp(from_node=scatter_recv_a)
    gather_send_b = GatherSendOp(from_node=scatter_recv_b)
    gather_recv_x_plus_one = GatherRecvOp(from_node=from_node, to_node=to_node,
                                          send_node=gather_send_a)
    return scatter_send_x, scatter_recv_a, scatter_recv_b, \
        gather_send_a, gather_send_b, gather_recv_x_plus_one
