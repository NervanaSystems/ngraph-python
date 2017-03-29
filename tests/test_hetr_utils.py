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
from ngraph.util.hetr_utils import comm_path_exists, update_comm_deps, find_recvs, \
    create_send_recv_graph, create_scatter_gather_graph
from ngraph.op_graph.comm_nodes import SendOp, ScatterSendOp, GatherSendOp
from ngraph.op_graph.comm_nodes import RecvOp, ScatterRecvOp, GatherRecvOp
import ngraph as ng


def test_find_recvs():
    z, recv_x, recv_x_plus_one, send_x, x_plus_one, from_node, send_x_plus_one = \
        create_send_recv_graph()

    assert set([recv_x]) == set(find_recvs(x_plus_one))
    assert set([recv_x]) == set(find_recvs(recv_x))
    assert len(find_recvs(from_node)) == 0
    assert set([recv_x]) == set(find_recvs(send_x_plus_one))
    assert set([recv_x_plus_one, recv_x]) == set(find_recvs(recv_x_plus_one))
    assert set([recv_x_plus_one, recv_x]) == set(find_recvs(z))


def test_find_recvs_scatter_gather():
    scatter_send_x, scatter_recv_a, scatter_recv_b, gather_send_a, gather_send_b, \
        gather_recv_x_plus_one = create_scatter_gather_graph()

    assert set([scatter_recv_a]) == set(find_recvs(gather_send_a))
    assert set([scatter_recv_b]) == set(find_recvs(gather_send_b))
    assert len(find_recvs(scatter_send_x)) == 0
    assert set([gather_recv_x_plus_one, scatter_recv_a]) == set(find_recvs(gather_recv_x_plus_one))
    assert set([scatter_recv_a]) == set(find_recvs(scatter_recv_a))


def test_comm_path_exists():
    axes = ng.make_axes([ng.make_axis(length=10, name='A'), ng.make_axis(length=15, name='B')])
    with ng.metadata(device=None, device_id=None, transformer=None, host_transformer=None):
        from_node = ng.placeholder(axes)
        to_node = ng.placeholder(axes)
    send_x = SendOp(from_node=from_node)
    recv_x = RecvOp(to_node=to_node, send_node=send_x)

    with ng.metadata(device=None, device_id=None, transformer=None, host_transformer=None):
        x_plus_one = recv_x + 1

    assert comm_path_exists(recv_x, send_x)
    assert comm_path_exists(x_plus_one, send_x)


def test_comm_path_exists_scatter_gather():
    scatter_send_x, scatter_recv_a, scatter_recv_b, gather_send_a, gather_send_b, \
        gather_recv_x_plus_one = create_scatter_gather_graph()

    assert comm_path_exists(scatter_recv_a, scatter_send_x)
    assert comm_path_exists(gather_recv_x_plus_one, gather_send_a)
    assert comm_path_exists(gather_recv_x_plus_one, scatter_send_x)
    assert comm_path_exists(scatter_recv_b, scatter_send_x)
    assert not comm_path_exists(gather_recv_x_plus_one, gather_send_b)
    assert not comm_path_exists(gather_send_a, gather_recv_x_plus_one)


def test_update_comm_deps():
    with ng.metadata(transformer='cpu0'):
        z, recv_x, recv_x_plus_one, send_x, x_plus_one, from_node, send_x_plus_one = \
            create_send_recv_graph()
    update_comm_deps((z, send_x))
    assert recv_x_plus_one in z.all_deps


def test_update_comm_deps_scatter_gather():
    ax_a = ng.make_axis(length=10, name='A')
    ax_b = ng.make_axis(length=15, name='B')
    axes = ng.make_axes([ax_a, ax_b])

    parallel_metadata = dict(parallel=ax_b, device_id=(0, 1),
                             transformer=None, host_transformer=None, device=None)
    with ng.metadata(transformer='cpu0'):
        with ng.metadata(**parallel_metadata):
            from_node_a = ng.placeholder(axes)
            to_node_a = ng.placeholder(axes)
        scatter_send_x = ScatterSendOp(from_node=from_node_a, to_node=to_node_a)
        scatter_recv_a = ScatterRecvOp(to_node=to_node_a, send_node=scatter_send_x)
        with ng.metadata(**parallel_metadata):
            x_plus_one_a = scatter_recv_a + 1
        gather_send_x_plus_one_a = GatherSendOp(from_node=x_plus_one_a)

    with ng.metadata(transformer='cpu1'):
        with ng.metadata(**parallel_metadata):
            to_node_b = ng.placeholder(axes)
        scatter_recv_b = ScatterRecvOp(to_node=to_node_b, send_node=scatter_send_x)
        with ng.metadata(**parallel_metadata):
            x_plus_one_b = scatter_recv_b + 1
        gather_send_x_plus_one_b = GatherSendOp(from_node=x_plus_one_b)

    with ng.metadata(transformer='cpu0'):
        with ng.metadata(**parallel_metadata):
            gather_recv_x_plus_one_a = GatherRecvOp(from_node=from_node_a, to_node=to_node_a,
                                                    send_node=gather_send_x_plus_one_a)
            z_a = gather_recv_x_plus_one_a + 1

    update_comm_deps((scatter_send_x, gather_send_x_plus_one_a, z_a))
    update_comm_deps((gather_send_x_plus_one_b,))

    assert set([scatter_send_x]) == set(scatter_recv_a.all_deps)
    assert set([scatter_send_x, gather_send_x_plus_one_a]) == \
        set(gather_recv_x_plus_one_a.all_deps)


def test_scatter_gather_node_axes():
    ax_A = ng.make_axis(64)
    ax_B = ng.make_axis(128)
    ax_C = ng.make_axis(255)

    tests = [
        {
            'axes': ng.make_axes([ax_A]),
            'parallel_axis': ax_A,
            'slices': [[slice(0, 32, 1)], [slice(32, 64, 1)]],
            'device_id': (0, 1)
        },
        {
            'axes': ng.make_axes([ax_A, ax_B]),
            'parallel_axis': ax_A,
            'slices': [[slice(0, 21, 1), slice(None)],
                       [slice(21, 42, 1), slice(None)],
                       [slice(42, 64, 1), slice(None)]],
            'device_id': (0, 1, 2)
        },
        {
            'axes': ng.make_axes([ax_A, ax_B, ax_C]),
            'parallel_axis': ax_A,
            'slices': [[slice(0, 12, 1), slice(None), slice(None)],
                       [slice(12, 24, 1), slice(None), slice(None)],
                       [slice(24, 36, 1), slice(None), slice(None)],
                       [slice(36, 48, 1), slice(None), slice(None)],
                       [slice(48, 64, 1), slice(None), slice(None)]],
            'device_id': (0, 1, 2, 3, 4)
        },
        {
            'axes': ng.make_axes([ax_A, ax_B, ax_C]),
            'parallel_axis': ax_C,
            'slices': [[slice(None), slice(None), slice(0, 127, 1)],
                       [slice(None), slice(None), slice(127, 255, 1)]],
            'device_id': (0, 1)
        }
    ]

    for t in tests:
        from_node = ng.placeholder(axes=t['axes'])
        from_node.metadata['device'] = None
        from_node.metadata['device_id'] = t['device_id']
        from_node.metadata['transformer'] = None
        from_node.metadata['parallel'] = t['parallel_axis']
        from_node.metadata['host_transformer'] = None

        to_node = ng.placeholder(axes=t['axes'])
        to_node.metadata['device'] = None
        to_node.metadata['device_id'] = t['device_id']
        to_node.metadata['transformer'] = None
        to_node.metadata['parallel'] = t['parallel_axis']
        to_node.metadata['host_transformer'] = None

        gather_send_op = GatherSendOp(from_node=from_node)
        assert t['axes'] == gather_send_op.axes

        gather_recv_op = GatherRecvOp(from_node=from_node,
                                      to_node=to_node,
                                      send_node=gather_send_op)
        assert t['axes'] == gather_recv_op.axes
        assert t['slices'] == gather_recv_op.slices

        scatter_send_op = ScatterSendOp(from_node=from_node,
                                        to_node=to_node)
        assert t['axes'] == scatter_send_op.axes
        assert t['slices'] == scatter_send_op.slices

        scatter_recv_op = ScatterRecvOp(to_node=to_node,
                                        send_node=scatter_send_op)
        assert t['axes'] == scatter_recv_op.axes

# todo def test_clone_graph():
