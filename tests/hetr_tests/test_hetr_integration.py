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
from test_hetr_passes import check_device_assign_pass, check_communication_pass
import ngraph as ng
import ngraph.transformers as ngt
from ngraph.op_graph.comm_nodes import CPUQueueAllReduceOp
from ngraph.transformers.hetr.mpilauncher import Launcher
from multiprocessing import active_children
import threading
import time
import os
import subprocess


pytestmark = pytest.mark.hetr_only
STARTUP_TIME = 3

def test_singleton_device_id(transformer_factory):
    with ng.metadata(device_id=(['1'])):
        x = ng.placeholder(())
    graph_ops = OrderedSet([x])

    graph_op_metadata = {op: list() for op in graph_ops}
    graph_op_metadata[x] = ["cpu", '1']

    check_device_assign_pass("cpu", "0", graph_op_metadata, graph_ops)


def test_scatter_gather_graph(transformer_factory):
    # Build the graph
    W = ng.make_axis(length=6, name='width')

    with ng.metadata(device_id='0'):
        x = ng.placeholder(())
        z = ng.placeholder(())

    with ng.metadata(device_id=('0', '1'), parallel=W):
        y = ng.placeholder(())

    x_plus_z = x + z  # Does not create a recv node
    x_plus_y = x + y  # creates a gather recv node

    # Build the graph metadata
    graph_ops = OrderedSet([x, y, z, x_plus_z, x_plus_y])

    graph_op_metadata = {op: list() for op in graph_ops}
    graph_op_metadata[x] = ["cpu", '0']
    graph_op_metadata[z] = ["cpu", '0']
    graph_op_metadata[y] = ["cpu", ('0', '1')]
    graph_op_metadata[x_plus_z] = ["cpu", '0']
    graph_op_metadata[x_plus_y] = ["cpu", '0']

    check_device_assign_pass("cpu", "0", graph_op_metadata, graph_ops)

    check_communication_pass(
        ops_to_transform=graph_ops,
        expected_recv_nodes=[x_plus_y])


ax_A = ng.make_axis(4)
ax_B = ng.make_axis(8)
ax_C = ng.make_axis(12)
ax_D = ng.make_axis(24)


@pytest.mark.multi_device
@pytest.mark.parametrize('config', [
    {
        'axes': ng.make_axes([ax_A]),
        'device_id': ('0', '1'),
        'parallel_axis': ax_A,
    },
    {
        'axes': ng.make_axes([ax_A, ax_B]),
        'device_id': ('0', '1', '2', '3'),
        'parallel_axis': ax_A,
    },
])
def test_distributed_plus_one(transformer_factory, hetr_device, config):
    device_id = config['device_id']
    axes = config['axes']
    parallel_axis = config['parallel_axis']

    if hetr_device == 'gpu':
        os.environ["HETR_SERVER_GPU_NUM"] = str(len(device_id))
        pytest.skip('gpu test is not supported now.')

    with ng.metadata(device=hetr_device):
        x = ng.placeholder(axes=axes)
        with ng.metadata(device_id=device_id, parallel=parallel_axis):
            x_plus_one = x + 1

    np_x = np.random.randint(100, size=axes.lengths)
    with ExecutorFactory() as ex:
        computation = ex.executor(x_plus_one, x)
        res = computation(np_x)
        np.testing.assert_array_equal(res, np_x + 1)


@pytest.mark.multi_device
@pytest.mark.parametrize('config', [
    {
        'axes_x': ng.make_axes([ax_A, ax_B]),
        'axes_w': ng.make_axes([ax_C, ax_A]),
        'device_id': ('0', '1'),
        'parallel_axis': ax_B,
    },
    {
        'axes_x': ng.make_axes([ax_A, ax_B]),
        'axes_w': ng.make_axes([ax_C, ax_A]),
        'device_id': ('0', '1', '2', '3'),
        'parallel_axis': ax_B,
    },
])
def test_distributed_dot(transformer_factory, hetr_device, config):
    device_id = config['device_id']
    axes_x = config['axes_x']
    axes_w = config['axes_w']
    parallel_axis = config['parallel_axis']

    if hetr_device == 'gpu':
        os.environ["HETR_SERVER_GPU_NUM"] = str(len(device_id))
        pytest.skip('gpu test is not supported now.')

    with ng.metadata(device=hetr_device):
        x = ng.placeholder(axes=axes_x)
        w = ng.placeholder(axes=axes_w)
        with ng.metadata(device_id=device_id, parallel=parallel_axis):
            dot = ng.dot(w, x)

    np_x = np.random.randint(100, size=axes_x.lengths)
    np_weight = np.random.randint(100, size=axes_w.lengths)
    with ExecutorFactory() as ex:
        computation = ex.executor(dot, x, w)
        res = computation(np_x, np_weight)
        np.testing.assert_array_equal(res, np.dot(np_weight, np_x))

@pytest.mark.multi_device
@pytest.mark.parametrize('config', [
    {
        'axes': ng.make_axes([ax_A]),
        'device_id': ('0', '1'),
        'parallel_axis': ax_A,
    },
    {
        'axes': ng.make_axes([ax_A, ax_B]),
        'device_id': ('0', '1', '2', '3'),
        'parallel_axis': ax_A,
    },
])
def test_distributed_plus_two(transformer_factory, hetr_device, config):
    device_id = config['device_id']
    axes = config['axes']
    parallel_axis = config['parallel_axis']

    if hetr_device == 'gpu':
        os.environ["HETR_SERVER_GPU_NUM"] = str(len(device_id))
        pytest.skip('gpu test is not supported now.')

    with ng.metadata(device=hetr_device):
        x = ng.placeholder(axes=axes)
        with ng.metadata(device_id=('0', '1'), parallel=parallel_axis):
            x_plus_one = x + 1
        x_plus_two = x_plus_one + 1

    np_x = np.random.randint(100, size=axes.lengths)
    with ExecutorFactory() as ex:
        computation = ex.executor(x_plus_two, x)
        res = computation(np_x)
        np.testing.assert_array_equal(res, np_x + 2)


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

            from ngraph.frontends.common.utils import conv_output_dim
            M = conv_output_dim(D, T, pad_d, str_d)
            P = conv_output_dim(H, R, pad_h, str_h)
            Q = conv_output_dim(W, S, pad_w, str_w)

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

        @property
        def has_side_effects(self):
            return True

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
        'axes': ng.make_axes([ax_A, ax_B]),
        'device_id': ('0', '1'),
        'parallel_axis': ax_B,
    },
    {
        'axes': ng.make_axes([ax_A, ax_B, ax_C]),
        'device_id': ('0', '1'),
        'parallel_axis': ax_A,
    },
    {
        'axes': ng.make_axes([ax_A, ax_B, ax_C]),
        'device_id': ('0', '1'),
        'parallel_axis': ax_B,
    },
    {
        'axes': ng.make_axes([ax_A, ax_B, ax_C]),
        'device_id': ('0', '1'),
        'parallel_axis': ax_C,
    },
    {
        'axes': ng.make_axes([ax_A, ax_B, ax_C, ax_D]),
        'device_id': ('0', '1'),
        'parallel_axis': ax_B,
    },
    {
        'axes': ng.make_axes([ax_A, ax_B, ax_C, ax_D]),
        'device_id': ('0', '1', '2'),
        'parallel_axis': ax_C,
    },
    {
        'axes': ng.make_axes([ax_A, ax_B, ax_C, ax_D]),
        'device_id': ('0', '1', '2'),
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

    os.environ["HETR_SERVER_GPU_NUM"] = str(len(t['device_id']))

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


class ClosingHetrServers():
    def __init__(self, ports):
        self.processes = []
        for p in ports:
            hetr_server = os.path.dirname(os.path.realpath(__file__)) +\
                "/../../ngraph/transformers/hetr/hetr_server.py"
            command = ["python", hetr_server, "-p", p]
            try:
                proc = subprocess.Popen(command)
                self.processes.append(proc)
            except Exception as e:
                print(e)
            time.sleep(STARTUP_TIME)

    def close(self):
        for p in self.processes:
            p.terminate()
        time.sleep(STARTUP_TIME)
        for p in self.processes:
            p.kill()
        for p in self.processes:
            p.wait()


def test_rpc_transformer():
    from ngraph.transformers.hetr.rpc_client import RPCTransformerClient
    rpc_client_list = list()
    port_list = ['50111', '50112']
    num_procs = len(port_list)

    with closing(ClosingHetrServers(port_list)):
        for p in range(num_procs):
            rpc_client_list.append(RPCTransformerClient('cpu' + str(p), port_list[p]))
            np.testing.assert_equal(rpc_client_list[p].initialized, True)


def test_mpilauncher():
    port_list = ['51111', '51112']
    num_procs = len(port_list)
    os.environ["HETR_SERVER_GPU_NUM"] = str(num_procs)

    mpilauncher = Launcher(port_list)
    mpilauncher.launch()

    # Check if process has launched
    assert mpilauncher.mpirun_proc.poll() is None

    mpilauncher.close()

    # Check if process has completed
    assert mpilauncher.mpirun_proc.poll() is not None
