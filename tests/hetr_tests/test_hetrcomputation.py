# ----------------------------------------------------------------------------
# copyright 2016 Nervana Systems Inc.
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
import numpy as np
import pytest
from contextlib import closing
from ngraph.testing import ExecutorFactory
from orderedset import OrderedSet
import ngraph as ng
import ngraph.transformers as ngt
from ngraph.transformers.passes.hetrpasses import DeviceAssignPass, \
    CommunicationPass
from ngraph.op_graph.comm_nodes import CPUQueueAllReduceOp, \
    GPUCudaAllReduceOp
from multiprocessing import active_children, Process, Event, Queue
import threading
import time


pytestmark = pytest.mark.hetr_only


def check_device_assign_pass(default_device, default_device_id,
                             graph_op_metadata, graph_op=OrderedSet(), *args):
    """
    The Device assign pass should inject the metadata{device_id, device} as
    specified by the user for each op,
    if not specified then the default {device_id:0, device:cpu} should be
    inserted for each op.

    :param: default_device: string, the default device for each op,
            if not specified by user ex: "cpu"
    :param: default_device_id: string, the default device number for each op,
            if not specified by user ex: "0"
    :param: graph_op_metadata: dict, dictionary of list specifying  the expected
            metadata {device_id, device} for each op
    :param: graph_op: list of ops to do the graph traversal
    """
    with ExecutorFactory():
        expected_transformers = set()

        class MockHetr(object):

            def __init__(self):
                self.transformers = set()

            def register_transformer(self, transformer):
                self.transformers.add(transformer)

        hetr = MockHetr()
        obj = DeviceAssignPass(hetr, default_device, default_device_id)

        obj.do_pass(ops=graph_op)

        for op in graph_op_metadata.keys():
            assert op.metadata['device'] == graph_op_metadata[op][0]
            assert op.metadata['device_id'] == graph_op_metadata[op][1]
            if isinstance(graph_op_metadata[op][1], (list, tuple)):
                transformer = [graph_op_metadata[op][0] + str(i) for i in graph_op_metadata[op][1]]
            else:
                transformer = graph_op_metadata[op][0] + str(graph_op_metadata[op][1][0])
            assert op.metadata['transformer'] == transformer

            for device_id in graph_op_metadata[op][1]:
                expected_transformers.add(graph_op_metadata[op][0] + device_id)
        assert hetr.transformers == expected_transformers


def check_communication_pass(ops_to_transform, expected_recv_nodes):
    """
    The communication pass should insert send/recv nodes wherever
    the metadata[transformer] differs between nodes.
    This checks that the recv nodes are inserted in the right place, and counts
    that the expected number of send
    nodes are found.

    :param ops_to_transform: list of ops to do the garph traversal
    :param expected_recv_nodes: lits of ops where receive nodes are expected to
           be inserted after the communication pass
    """
    with ExecutorFactory():
        send_nodes = OrderedSet()
        obj = CommunicationPass(send_nodes)
        obj.do_pass(ops_to_transform)

        op_list_instance_type = list()
        num_expected_sendnodes = len(expected_recv_nodes)

        # Count if the communication pass inserted the expected number of send nodes
        assert num_expected_sendnodes == len(send_nodes)

        # verify if Recv nodes are inserted in the right place
        for op in expected_recv_nodes:
            for each_arg in op.args:
                op_list_instance_type.append(type(each_arg))

            if (ng.op_graph.comm_nodes.CPUQueueRecvOp in op_list_instance_type or
                ng.op_graph.comm_nodes.CPUQueueGatherRecvOp in op_list_instance_type or
                    ng.op_graph.comm_nodes.CPUQueueScatterRecvOp in
                    op_list_instance_type) is False:
                assert False
            del op_list_instance_type[:]


def test_hetr_graph_passes(transformer_factory):

    # Build the graph
    with ng.metadata(device_id='1'):
        x = ng.placeholder(())

    y = ng.placeholder(())
    x_plus_y = x + y

    # Build the graph metadata
    graph_ops = OrderedSet([x_plus_y, x, y])

    graph_op_metadata = {op: list() for op in graph_ops}
    graph_op_metadata[x] = ["cpu", '1']
    graph_op_metadata[y] = ["cpu", '0']
    graph_op_metadata[x_plus_y] = ["cpu", '0']

    # Run the hetr passes one by one, and verify they did the expected things to the graph
    check_device_assign_pass("cpu", "0", graph_op_metadata, graph_ops)
    check_communication_pass(ops_to_transform=graph_ops,
                             expected_recv_nodes=[x_plus_y])


def test_distributed_graph_plus_one(transformer_factory):
    H = ng.make_axis(length=4, name='height')
    W = ng.make_axis(length=6, name='width')
    x = ng.placeholder(axes=[H, W])
    with ng.metadata(device_id=('1', '2'), parallel=W):
        x_plus_one = x + 1

    np_x = np.random.randint(100, size=[H.length, W.length])
    with ExecutorFactory() as ex:
        computation = ex.executor(x_plus_one, x)
        res = computation(np_x)
        np.testing.assert_array_equal(res, np_x + 1)


def test_distributed_dot(transformer_factory):
    H = ng.make_axis(length=4, name='height')
    N = ng.make_axis(length=8, name='batch')
    weight = ng.make_axis(length=2, name='weight')
    x = ng.placeholder(axes=[H, N])
    w = ng.placeholder(axes=[weight, H])
    with ng.metadata(device_id=('1', '2'), parallel=N):
        dot = ng.dot(w, x)

    np_x = np.random.randint(100, size=[H.length, N.length])
    np_weight = np.random.randint(100, size=[weight.length, H.length])
    with ExecutorFactory() as ex:
        computation = ex.executor(dot, x, w)
        res = computation(np_x, np_weight)
        np.testing.assert_array_equal(res, np.dot(np_weight, np_x))


def test_distributed_graph_plus_two(transformer_factory):
    H = ng.make_axis(length=4, name='height')
    W = ng.make_axis(length=6, name='width')
    x = ng.placeholder(axes=[H, W])
    with ng.metadata(device_id=('1', '2'), parallel=W):
        x_plus_one = x + 1
    x_plus_two = x_plus_one + 1

    np_x = np.random.randint(100, size=[H.length, W.length])
    with ExecutorFactory() as ex:
        computation = ex.executor(x_plus_two, x)
        res = computation(np_x)
        np.testing.assert_array_equal(res, np_x + 2)


def test_singleton_device_id(transformer_factory):
    with ng.metadata(device_id=(['1'])):
        x = ng.placeholder(())
    graph_ops = OrderedSet([x])

    graph_op_metadata = {op: list() for op in graph_ops}
    graph_op_metadata[x] = ["cpu", '1']

    check_device_assign_pass("cpu", "0", graph_op_metadata, graph_ops)


def test_from_device(transformer_factory):
    with ng.metadata(device_id='1'):
        x = ng.placeholder(())
    x_plus_one = x + 1

    with ExecutorFactory() as ex:
        computation = ex.executor(x_plus_one, x)
        for i in [10, 20, 30]:
            assert computation(i) == i + 1


def test_to_device(transformer_factory):
    x = ng.placeholder(())
    with ng.metadata(device_id='1'):
        x_plus_one = x + 1

    with ExecutorFactory() as ex:
        computation = ex.executor(x_plus_one, x)
        for i in [10, 20, 30]:
            assert computation(i) == i + 1


def test_to_and_from_device(transformer_factory):
    x = ng.placeholder(())
    with ng.metadata(device_id='1'):
        x_plus_one = x + 1
    x_plus_two = x_plus_one + 1

    with ExecutorFactory() as ex:
        computation = ex.executor(x_plus_two, x)
        for i in [10, 20, 30]:
            assert computation(i) == i + 2


def test_computation_return_list(transformer_factory):
    with ng.metadata(device_id='1'):
        x = ng.placeholder(())
    x_plus_one = x + 1
    x_plus_two = x + 2
    x_mul_three = x * 3

    with ExecutorFactory() as ex:
        computation = ex.executor([x_plus_one, x_plus_two, x_mul_three], x)
        for i in [10, 20, 30]:
            assert computation(i) == (i + 1, i + 2, i * 3)


def test_scatter_gather_graph(transformer_factory):
    # Build the graph
    W = ng.make_axis(length=6, name='width')

    with ng.metadata(device_id='0'):
        x = ng.placeholder(())
        z = ng.placeholder(())

    with ng.metadata(device_id=('1', '2'), parallel=W):
        y = ng.placeholder(())

    x_plus_z = x + z  # Does not create a recv node
    x_plus_y = x + y  # creates a gather recv node

    # Build the graph metadata
    graph_ops = OrderedSet([x, y, z, x_plus_z, x_plus_y])

    graph_op_metadata = {op: list() for op in graph_ops}
    graph_op_metadata[x] = ["cpu", '0']
    graph_op_metadata[z] = ["cpu", '0']
    graph_op_metadata[y] = ["cpu", ('1', '2')]
    graph_op_metadata[x_plus_z] = ["cpu", '0']
    graph_op_metadata[x_plus_y] = ["cpu", '0']

    check_device_assign_pass("cpu", "0", graph_op_metadata, graph_ops)

    check_communication_pass(
        ops_to_transform=graph_ops,
        expected_recv_nodes=[x_plus_y])


@pytest.mark.hetr_gpu_only
def test_gpu_send_and_recv():
    # put x+1 on cpu numpy
    with ng.metadata(device='cpu'):
        x = ng.placeholder(())
        x_plus_one = x + 1
    # put x+2 on gpu numpy
    with ng.metadata(device='gpu'):
        x_plus_two = x_plus_one + 1

    with ExecutorFactory() as ex:
        computation = ex.executor(x_plus_two, x)
        for i in [10, 20, 30]:
            assert computation(i) == i + 2

    # put x+1 on gpu numpy
    with ng.metadata(device='gpu'):
        x = ng.placeholder(())
        x_plus_one = x + 1
    # put x+2 on cpu numpy
    with ng.metadata(device='cpu'):
        x_plus_two = x_plus_one + 1

    with ExecutorFactory() as ex:
        computation = ex.executor(x_plus_two, x)
        for i in [10, 20, 30]:
            assert computation(i) == i + 2


def test_recvop_axes_using_dot(transformer_factory):
    x_value = np.array([[1],
                        [2]])
    w_value = np.array([[-1, 1]])

    A1 = ng.make_axis(length=1)
    A2 = ng.make_axis(length=2)
    A3 = ng.make_axis(length=2)

    x = ng.placeholder([A2, A1])
    w = ng.variable([A1, A3], initial_value=w_value)

    with ng.metadata(device_id='1'):
        result = ng.dot(x, w)

    with ExecutorFactory() as ex:
        computation = ex.executor(result, x, w)
        val_ng = computation(x_value, w_value)
        val_np = np.dot(x_value, w_value)
        assert ng.testing.allclose(val_ng, val_np)


def test_recvop_tensorupdate(transformer_factory):
    """
    The tensor (RecvOp_#_#) associated with the following conv op has two views:
    1) Non-flat view (e.g. RecvOp_#_#_1_1_1_1_4.shape=(1,1,1,1,4))
    2) Flat view (e.g. RecvOp_#_#_1_4.shape = (1,4))
    This test ensures that inside RecvOp code generation, the generated code
    should make sure both views get updated (e.g. by using update_RecvOp_#_# API)
    In this test, ng.dot operation tends to use the flat view (i.e. RecvOp_#_#_1_4)
    And previously RecvOp with RecvOp_#_#_1_1_1_1_4 = recv_from_send(send_id) failed
    to update both two views (i.e. flat and non-flat view of the same buffer/tensor)
    """
    class ConvParams(object):

        def __init__(self, C=1, N=1, K=1, D=1, H=1, W=1, T=1, R=1, S=1,
                     pad_d=0, pad_h=0, pad_w=0,
                     str_d=1, str_h=1, str_w=1):

            from ngraph.frontends.neon.layer import output_dim
            M = output_dim(D, T, pad_d, str_d)
            P = output_dim(H, R, pad_h, str_h)
            Q = output_dim(W, S, pad_w, str_w)

            self.dimO = (K, M, P, Q, N)
            self.dimI = (C, D, H, W, N)
            self.dimF = (C, T, R, S, K)

            self.conv_params = dict(
                pad_d=pad_d, pad_h=pad_h, pad_w=pad_w,
                str_d=str_d, str_h=str_h, str_w=str_w,
                dil_d=1, dil_h=1, dil_w=1
            )

            self.batch_axis = ng.make_axis(name='N', length=N)

            self.ax_i = ng.make_axes([
                ng.make_axis(name='C', length=C),
                ng.make_axis(name='D', length=D),
                ng.make_axis(name='H', length=H),
                ng.make_axis(name='W', length=W),
                self.batch_axis
            ])

            self.ax_f = ng.make_axes([
                ng.make_axis(name='C', length=C),
                ng.make_axis(name='D', length=T),
                ng.make_axis(name='H', length=R),
                ng.make_axis(name='W', length=S),
                ng.make_axis(name='K', length=K),
            ])

            self.ax_o = ng.make_axes([
                ng.make_axis(name='C', length=K),
                ng.make_axis(name='D', length=M),
                ng.make_axis(name='H', length=P),
                ng.make_axis(name='W', length=Q),
                self.batch_axis
            ])

    # Layer 1, using convolutation introduces multi/flatten view of tensors
    cf = ConvParams(C=2, N=4, K=1, H=2, W=2, R=2, S=2)

    inputs = ng.placeholder(axes=cf.ax_i)
    filters = ng.placeholder(axes=cf.ax_f)

    # randomly initialize
    from ngraph.testing import RandomTensorGenerator
    rng = RandomTensorGenerator(0, np.float32)
    # put value 1 into inputs/filters for conv
    input_value = rng.uniform(1, 1, cf.ax_i)
    filter_value = rng.uniform(1, 1, cf.ax_f)

    conv = ng.convolution(cf.conv_params, inputs, filters, axes=cf.ax_o)

    # Layer 2, using dot to ensure recv_op.axes == send_op.axes
    from ngraph.frontends.neon import UniformInit
    # put value 1 into weights for dot
    init_uni = UniformInit(1, 1)
    W_A = ng.make_axis(length=2)
    w_axes = ng.make_axes(W_A) + conv.axes.feature_axes()
    w = ng.variable(axes=w_axes, initial_value=init_uni)

    with ng.metadata(device_id='1'):
        dot = ng.dot(w, conv)

    with ExecutorFactory() as ex:
        dot_comp = ex.executor(dot, filters, inputs)
        dot_val = dot_comp(filter_value, input_value)

    np.testing.assert_array_equal(dot_val, [[8., 8., 8., 8.],
                                            [8., 8., 8., 8.]])


def test_terminate_op(transformer_factory):

    class TerminateOp(ng.Op):

        def __init__(self, **kwargs):
            super(TerminateOp, self).__init__(**kwargs)

    baseline = active_children()
    termOp = TerminateOp()
    assert len(baseline) == 0
    with ExecutorFactory() as ex:
        comp = ex.executor(termOp)
        assert len(active_children()) == 1
        with pytest.raises(RuntimeError):
            comp()
        assert len(active_children()) == 1
    assert len(active_children()) == len(baseline)


def test_process_leak(transformer_factory):
    baseline = active_children()
    with ng.metadata(device_id=('2')):
        x = ng.constant(2)
    assert len(active_children()) == 0
    with ExecutorFactory() as ex:
        comp = ex.executor(x)
        assert len(active_children()) == 1
        comp()
        assert len(active_children()) == 2
    assert len(active_children()) == len(baseline)


ax_A = ng.make_axis(4)
ax_B = ng.make_axis(6)
ax_C = ng.make_axis(12)
ax_D = ng.make_axis(24)


@pytest.mark.hetr_gpu_only
@pytest.mark.parametrize('config', [
    {
        'axes': ng.make_axes([ax_A]),
        'device_id': ('0'),
        'parallel_axis': ax_A,
    },
    {
        'axes': ng.make_axes([ax_A]),
        'device_id': ('0', '1'),
        'parallel_axis': ax_A,
    },
    {
        'axes': ng.make_axes([ax_A, ax_B]),
        'device_id': ('0', '1'),
        'parallel_axis': ax_A,
    },
    {
        'axes': ng.make_axes([ax_A]),
        'device_id': ('1', '2'),
        'parallel_axis': ax_A,
    },
    {
        'axes': ng.make_axes([ax_A, ax_B]),
        'device_id': ('1', '2'),
        'parallel_axis': ax_A,
    },
    {
        'axes': ng.make_axes([ax_A, ax_B]),
        'device_id': ('1', '2'),
        'parallel_axis': ax_B,
    },
    {
        'axes': ng.make_axes([ax_A, ax_B, ax_C]),
        'device_id': ('1', '2'),
        'parallel_axis': ax_A,
    },
    {
        'axes': ng.make_axes([ax_A, ax_B, ax_C]),
        'device_id': ('1', '2'),
        'parallel_axis': ax_B,
    },
    {
        'axes': ng.make_axes([ax_A, ax_B, ax_C]),
        'device_id': ('1', '2'),
        'parallel_axis': ax_C,
    },
    {
        'axes': ng.make_axes([ax_A, ax_B, ax_C, ax_D]),
        'device_id': ('1', '2'),
        'parallel_axis': ax_B,
    },
    {
        'axes': ng.make_axes([ax_A, ax_B, ax_C, ax_D]),
        'device_id': ('1', '2', '3'),
        'parallel_axis': ax_C,
    },
    {
        'axes': ng.make_axes([ax_A, ax_B, ax_C, ax_D]),
        'device_id': ('1', '2', '3'),
        'parallel_axis': ax_D,
    },
])
def test_gpu_graph(config):
    pytest.xfail("Multi-GPU testing not enabled yet")

    if 'gpu' not in ngt.transformer_choices():
        pytest.skip('GPUTransformer not available!')

    t = config
    with ng.metadata(device='gpu'):
        x = ng.placeholder(axes=t['axes'])

    with ng.metadata(device='gpu', device_id=t['device_id'], parallel=t['parallel_axis']):
        x_plus_one = x + 1

    with ng.metadata(device='gpu'):
        x_plus_two = x_plus_one + 1

    np_x = np.random.randint(100, size=t['axes'].full_lengths)
    with closing(ngt.make_transformer_factory('hetr')()) as transformer:
        computation = transformer.computation(x_plus_two, x)
        res = computation(np_x)
        np.testing.assert_array_equal(res, np_x + 2)


@pytest.mark.parametrize('config', [
    {
        'device_id': ('0', '1', '2', '3'),
        'x_input': [6, 3, 9, 10],
        'func': 'mean',
        'results': [7, 7, 7, 7],
    },
    {
        'device_id': ('0', '1', '2', '3', '4', '5', '6', '7'),
        'x_input': [5, 6, 11, 13, 2, 3, 5, 7],
        'func': 'sum',
        'results': [52, 52, 52, 52, 52, 52, 52, 52],
    },
])
def test_allreduce_cpu_op(config):
    class myThread(threading.Thread):
        def __init__(self, y):
            threading.Thread.__init__(self)
            self.y = y

        def run(self):
            with closing(ngt.make_transformer_factory('cpu')()) as t:
                comp = t.computation(self.y)
                self.result = comp()

        def get_result(self):
            self.join()
            return self.result

    c = config
    x = list()
    y = list()
    thread = list()
    results = list()

    with ng.metadata(device='cpu',
                     device_id=c['device_id'],
                     transformer='None',
                     host_transformer='None'):
        for i in c['x_input']:
            x.append(ng.constant(i))

    for i in range(len(c['device_id'])):
        ar_op = CPUQueueAllReduceOp(x[i], c['func'])
        if (i != 0):
            ar_op.idx = i
            ar_op._shared_queues = y[0].shared_queues
        y.append(ar_op)

    for i in range(len(c['device_id'])):
        thread.append(myThread(y[i]))
        thread[i].start()

    for i in range(len(c['device_id'])):
        results.append(thread[i].get_result())

    np.testing.assert_array_equal(results, c['results'])


@pytest.mark.hetr_gpu_only
@pytest.mark.parametrize('config', [
    {
        'device_id': (0, 1),
        'x_input': np.arange(24),
        'func': 'mean',
    },
    {
        'device_id': (0, 1),
        'x_input': np.arange(32),
        'func': 'sum',
    },
    {
        'device_id': (0, 1, 2, 3),
        'x_input': np.arange(48),
        'func': 'mean',
    },
    {
        'device_id': (0, 1, 2, 3),
        'x_input': np.arange(64),
        'func': 'sum',
    }
])
def test_allreduce_gpu_op(config):
    class myProcess(Process):
        def __init__(self, y, device_id, queue):
            Process.__init__(self)
            self.y = y
            self.device_id = device_id
            self.exit = Event()
            self.queue = queue

        def run(self):
            with closing(ngt.make_transformer_factory('gpu', device_id=self.device_id)()) as t:
                comp = t.computation(self.y)
                self.queue.put(comp())

            while not self.exit.is_set():
                time.sleep(0.1)

    pytest.xfail("Multi-GPU testing not enabled yet")

    if 'gpu' not in ngt.transformer_choices():
        pytest.skip('GPUTransformer not available!')

    c = config
    x = list()
    y = list()
    input_list = list()
    process_list = list()
    result_list = list()
    np_result_list = list()
    queue = Queue()

    with ng.metadata(device='gpu', device_id=c['device_id'],
                     transformer='None', host_transformer='None'):
        for i in range(len(c['device_id'])):
            x_input = c['x_input'] * (i + 1)
            x.append(ng.constant(x_input))
            input_list.append(x_input)

    for i in range(len(c['device_id'])):
        ar_op = GPUCudaAllReduceOp(x[i], c['func'])
        if (i != 0):
            ar_op._shared_queues = y[0].shared_queues
        y.append(ar_op)

    if c['func'] == 'mean':
        np_result = np.mean(input_list, axis=0)
    elif c['func'] == 'sum':
        np_result = np.sum(input_list, axis=0)

    for i, d in enumerate(c['device_id']):
        process_list.append(myProcess(y[i], d, queue))
        process_list[i].start()

    for p in reversed(process_list):
        np_result_list.append(np_result)
        result_list.append(queue.get())
        p.exit.set()
        p.join()

    np.testing.assert_array_equal(result_list, np_result_list)
