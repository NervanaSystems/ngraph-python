# ******************************************************************************
# Copyright 2017-2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************
import numpy as np
import pytest
from contextlib import closing
from ngraph.testing import ExecutorFactory
import ngraph as ng
import ngraph.transformers as ngt
from ngraph.transformers.hetr.mpilauncher import MPILauncher
import time
import os
import subprocess
import tempfile
import random


pytestmark = pytest.mark.hetr_only
STARTUP_TIME = 3


ax_A = ng.make_axis(4)
ax_B = ng.make_axis(8)
ax_C = ng.make_axis(12)
ax_D = ng.make_axis(24)


@pytest.mark.multi_device
@pytest.mark.parametrize('config', [
    {
        'device_id': ('0', '1'),
    },
    {
        'device_id': ('0', '1', '2', '3'),
    },
])
def test_broadcast_scalar(hetr_device, config):
    if hetr_device == 'gpu':
        pytest.skip('gpu communication broadcast op is not supported.')
    device_id = config['device_id']
    x = ng.placeholder(())
    y = ng.placeholder(())
    with ng.metadata(device_id=device_id, parallel=ax_A):
        x_plus_y = x + y

    with closing(ngt.make_transformer_factory('hetr', device=hetr_device)()) as transformer:
        computation = transformer.computation(x_plus_y, x, y)
        res = computation(1, 2)
        np.testing.assert_array_equal(res, 3)


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
def test_distributed_plus_one(hetr_device, config):
    device_id = config['device_id']
    axes = config['axes']
    parallel_axis = config['parallel_axis']

    with ng.metadata(device=hetr_device):
        x = ng.placeholder(axes=axes)
        with ng.metadata(device_id=device_id, parallel=parallel_axis):
            x_plus_one = x + 1

    np_x = np.random.randint(100, size=axes.lengths)
    with closing(ngt.make_transformer_factory('hetr', device=hetr_device)()) as transformer:
        computation = transformer.computation(x_plus_one, x)
        res = computation(np_x)
        np.testing.assert_array_equal(res, np_x + 1)


@pytest.mark.multi_device
@pytest.mark.parametrize('config', [
    {
        'axes_x': ng.make_axes([ax_B, ax_A]),
        'axes_w': ng.make_axes([ax_A, ax_C]),
        'device_id': ('0', '1'),
        'parallel_axis': ax_B,
    },
    {
        'axes_x': ng.make_axes([ax_B, ax_A]),
        'axes_w': ng.make_axes([ax_A, ax_C]),
        'device_id': ('0', '1', '2', '3'),
        'parallel_axis': ax_B,
    },
])
def test_distributed_dot(hetr_device, config):
    if hetr_device == 'gpu':
        pytest.xfail("Intermittent failure on jenkins for mgpu")
    device_id = config['device_id']
    axes_x = config['axes_x']
    axes_w = config['axes_w']
    parallel_axis = config['parallel_axis']

    np_weight = np.ones(axes_w.lengths)
    with ng.metadata(device=hetr_device):
        x = ng.placeholder(axes=axes_x)
        with ng.metadata(device_id=device_id, parallel=parallel_axis):
            w = ng.variable(axes=axes_w, initial_value=np_weight)
            dot = ng.dot(x, w)

    np_x = np.random.randint(100, size=axes_x.lengths)
    with closing(ngt.make_transformer_factory('hetr',
                 device=hetr_device)()) as transformer:
        computation = transformer.computation(dot, x)
        res = computation(np_x)
        np.testing.assert_array_equal(res, np.dot(np_x, np_weight))


@pytest.mark.multi_device
def test_multi_computations(hetr_device):
    if hetr_device == 'gpu':
        pytest.xfail("enable after gpu exgraph")
    axes_x = ng.make_axes([ax_A, ax_B])
    x = ng.placeholder(axes=axes_x)
    y = ng.placeholder(())
    with ng.metadata(device_id=('0', '1'), parallel=ax_A):
        f = x ** 2
        out = y - ng.mean(f, out_axes=())

    np_x = np.random.randint(10, size=axes_x.lengths)
    np_y = np.random.randint(10)
    with closing(ngt.make_transformer_factory('hetr', device=hetr_device)()) as t:
        comp = t.computation(out, x, y)
        another_comp = t.computation(f, x)

        res_comp = comp(np_x, np_y)
        res_another_comp = another_comp(np_x)
        ref_comp = np_y - np.mean(np_x**2)
        np.testing.assert_array_equal(res_comp, ref_comp)
        np.testing.assert_array_equal(res_another_comp, np_x**2)


@pytest.mark.multi_device
@pytest.mark.parametrize('config', [
    {
        'axes': ng.make_axes([ax_A]),
        'device_id': ('0', '1'),
        'parallel_axis': ax_A,
    },
])
def test_repeat_computation(hetr_device, config):
    if hetr_device == 'gpu':
        pytest.xfail("enable after gpu exgraph")
    device_id = config['device_id']
    axes = config['axes']
    parallel_axis = config['parallel_axis']

    with ng.metadata(device=hetr_device):
        x = ng.placeholder(axes=axes)
        with ng.metadata(device_id=device_id, parallel=parallel_axis):
            x_plus_one = x + 1

        np_x = np.random.randint(100, size=axes.lengths)
        with closing(ngt.make_transformer_factory('hetr', device=hetr_device)()) as transformer:
            comp = transformer.computation(x_plus_one, x)
            comp2 = transformer.computation(x_plus_one, x)

            res = comp(np_x)
            np.testing.assert_array_equal(res, np_x + 1)

            res2 = comp2(np_x)
            np.testing.assert_array_equal(res2, np_x + 1)


@pytest.mark.multi_device
def test_comm_broadcast_op(hetr_device):
    if hetr_device == 'gpu':
        pytest.skip('gpu communication broadcast op is not supported.')
    H = ng.make_axis(length=4, name='height')
    N = ng.make_axis(length=8, name='batch')
    weight = ng.make_axis(length=2, name='weight')
    x = ng.placeholder(axes=[N, H])
    # w will be broadcasted to devices
    w = ng.placeholder(axes=[H, weight])
    with ng.metadata(device=hetr_device, device_id=('0', '1'), parallel=N):
        dot = ng.dot(x, w)

    np_x = np.random.randint(100, size=[N.length, H.length])
    np_weight = np.random.randint(100, size=[H.length, weight.length])
    with closing(ngt.make_transformer_factory('hetr', device=hetr_device)()) as transformer:
        computation = transformer.computation(dot, x, w)
        res = computation(np_x, np_weight)
        np.testing.assert_array_equal(res, np.dot(np_x, np_weight))


@pytest.mark.multi_device
def test_reduce_scalar(hetr_device):
    """
    A scalar is produced by sum() on each worker
    in this case, should be mean reduced before being returned
    """
    if hetr_device == 'gpu':
        pytest.xfail("gather/reduce work-around for gpus does not choose between mean or sum,\
        it uses only the value on the first device and ignores the values on other devices")

    N = ng.make_axis(length=8, name='batch')
    x = ng.placeholder(axes=[N])
    with ng.metadata(device=hetr_device, device_id=('0', '1'), parallel=N):
        out = ng.sum(x)

    np_x = np.random.randint(100, size=[N.length])
    with closing(ngt.make_transformer_factory('hetr', device=hetr_device)()) as transformer:
        computation = transformer.computation(out, x)
        res = computation(np_x)

        # gather returns one element per worker
        np.testing.assert_array_equal(res, np.sum(np_x) / 2.)


@pytest.mark.multi_device
def test_reduce_vector(hetr_device):
    """
    A whole vector is produced on each worker and should be reduced
    before being returned, but not along its axes since it
    does not have the parallel axis in its axes
    """
    if hetr_device == 'gpu':
        pytest.xfail("broadcast communication ops not yet supported on gpus")

    H = ng.make_axis(length=4, name='height')
    N = ng.make_axis(length=8, name='batch')
    weight = ng.make_axis(length=2, name='weight')
    x = ng.placeholder(axes=[N, H])
    w = ng.placeholder(axes=[H, weight])
    with ng.metadata(device=hetr_device, device_id=('0', '1'), parallel=N):
        dot = ng.dot(x, w)
        out = ng.sum(dot, N)

    np_x = np.random.randint(100, size=[N.length, H.length])
    np_weight = np.random.randint(100, size=[H.length, weight.length])
    with closing(ngt.make_transformer_factory('hetr', device=hetr_device)()) as transformer:
        computation = transformer.computation(out, x, w)
        res = computation(np_x, np_weight)
        # TODO should the reduce infer a sum or mean?
        expected = np.sum(np.dot(np_x, np_weight), 0) / 2.
        np.testing.assert_array_equal(res, expected)


@pytest.mark.multi_device
def test_distributed_dot_parallel_second_axis(hetr_device):
    if hetr_device == 'gpu':
        pytest.xfail("Axes Layout needs to be fixed for GPUs after changes to make\
        parallel_axis the least contiguous axis for scatter/gather communication ops")

    H = ng.make_axis(length=6, name='height')
    N = ng.make_axis(length=8, name='batch')
    W1 = ng.make_axis(length=2, name='W1')
    W2 = ng.make_axis(length=4, name='W2')
    x = ng.placeholder(axes=[H, N])
    w2 = ng.placeholder(axes=[W2, W1])
    with ng.metadata(device=hetr_device, device_id=('0', '1'), parallel=N):
        w1 = ng.placeholder(axes=[W1, H])
        dot1 = ng.dot(w1, x).named("dot1")
    dot2 = ng.dot(w2, dot1).named("dot2")

    np_x = np.random.randint(100, size=[H.length, N.length])
    np_w1 = np.random.randint(100, size=[W1.length, H.length])
    np_w2 = np.random.randint(100, size=[W2.length, W1.length])
    with closing(ngt.make_transformer_factory('hetr', device=hetr_device)()) as transformer:
        computation = transformer.computation([dot2, dot1], x, w1, w2)
        res2, res1 = computation(np_x, np_w1, np_w2)
        np.testing.assert_array_equal(res1, np.dot(np_w1, np_x))
        np.testing.assert_array_equal(res2, np.dot(np_w2, np.dot(np_w1, np_x)))

        computation2 = transformer.computation([dot1, dot2], x, w1, w2)
        res1, res2 = computation2(np_x, np_w1, np_w2)
        np.testing.assert_array_equal(res1, np.dot(np_w1, np_x))
        np.testing.assert_array_equal(res2, np.dot(np_w2, np.dot(np_w1, np_x)))


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
def test_distributed_plus_two(hetr_device, config):
    device_id = config['device_id']
    axes = config['axes']
    parallel_axis = config['parallel_axis']

    with ng.metadata(device=hetr_device):
        x = ng.placeholder(axes=axes)
        with ng.metadata(device_id=device_id, parallel=parallel_axis):
            x_plus_one = x + 1
        x_plus_two = x_plus_one + 1

    np_x = np.random.randint(100, size=axes.lengths)
    with closing(ngt.make_transformer_factory('hetr', device=hetr_device)()) as transformer:
        computation = transformer.computation(x_plus_two, x)
        res = computation(np_x)
        np.testing.assert_array_equal(res, np_x + 2)


@pytest.mark.multi_device
@pytest.mark.parametrize('config', [
    {
        'axes': None,
    },
    {
        'axes': ng.make_axes([ax_A]),
    },
    {
        'axes': ng.make_axes([ax_A, ax_B]),
    },
])
def test_to_and_from_device(hetr_device, config):
    axes = config['axes']
    with ng.metadata(device=hetr_device):
        x = ng.placeholder(axes=axes) if axes else ng.placeholder(())
        with ng.metadata(device_id='1'):
            x_plus_one = x + 1
        x_plus_two = x_plus_one * 2

    np_x = np.random.randint(100, size=axes.lengths) if axes else random.random()
    with closing(ngt.make_transformer_factory('hetr', device=hetr_device)()) as transformer:
        computation = transformer.computation([x_plus_one, x_plus_two], x)
        res = computation(np_x)
        np.testing.assert_allclose(res[0], np_x + 1.0)
        np.testing.assert_allclose(res[1], (np_x + 1.0) * 2.0)


@pytest.mark.multi_device
@pytest.mark.parametrize('config', [
    {
        'axes': None,
    },
    {
        'axes': ng.make_axes([ax_A]),
    },
    {
        'axes': ng.make_axes([ax_A, ax_B]),
    },
])
def test_computation_return_list(hetr_device, config):
    axes = config['axes']
    with ng.metadata(device=hetr_device):
        x = ng.placeholder(axes=axes) if axes else ng.placeholder(())

        with ng.metadata(device_id='1'):
            x_plus_one = x + 1
        with ng.metadata(device_id='2'):
            x_plus_two = x_plus_one + 1
        with ng.metadata(device_id='3'):
            x_mul_two = x_plus_two * 2

    np_x = np.random.randint(100, size=axes.lengths) if axes else random.random()
    with closing(ngt.make_transformer_factory('hetr', device=hetr_device)()) as transformer:
        computation = transformer.computation([x_plus_one, x_plus_two, x_mul_two], x)
        res = computation(np_x)
        np.testing.assert_allclose(res[0], np_x + 1)
        np.testing.assert_allclose(res[1], np_x + 2)
        np.testing.assert_allclose(res[2], (np_x + 2) * 2)


@pytest.mark.hetr_gpu_only
def test_gpu_send_and_recv(hetr_device):
    pytest.xfail("GitHub issue: #2007, Unknown error - investigation is needed")
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


def test_recvop_axes_using_dot():
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
        ng.testing.assert_allclose(val_ng, val_np)


def test_recvop_tensorupdate():
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


class ClosingHetrServers():
    def __init__(self, ports):
        self.tmpfile = tempfile.NamedTemporaryFile(dir=os.path.dirname(os.path.realpath(__file__)),
                                                   delete=True)
        self.processes = []
        for p in ports:
            hetr_server = os.path.dirname(os.path.realpath(__file__)) +\
                "/../../ngraph/transformers/hetr/hetr_server.py"
            command = ["python", hetr_server, "-tf", self.tmpfile.name, "-p", p]
            try:
                proc = subprocess.Popen(command)
                self.processes.append(proc)
            except Exception as e:
                print(e)
        time.sleep(STARTUP_TIME)

    def close(self):
        for p in self.processes:
            p.terminate()
        for p in self.processes:
            p.kill()
        for p in self.processes:
            p.wait()
        self.tmpfile.close()


def test_rpc_transformer():
    pytest.xfail("Needs investigation-STARTUP_TIME is too large, needs to be over 5 seconds.")
    from ngraph.transformers.hetr.rpc_client import RPCTransformerClient
    rpc_client_list = list()
    port_list = ['50111', '50112']
    num_procs = len(port_list)

    with closing(ClosingHetrServers(port_list)):
        for p in range(num_procs):
            rpc_client_list.append(RPCTransformerClient('cpu' + str(p),
                                                        'localhost:' + port_list[p]))
            np.testing.assert_equal(rpc_client_list[p].is_trans_built, False)
            np.testing.assert_equal(rpc_client_list[p].transformer_type, 'cpu' + str(p))
            np.testing.assert_equal(rpc_client_list[p].server_address, 'localhost:' + port_list[p])
        for p in range(num_procs):
            rpc_client_list[p].build_transformer()
            np.testing.assert_equal(rpc_client_list[p].is_trans_built, True)
        for p in range(num_procs):
            rpc_client_list[p].close_transformer()
        for p in range(num_procs):
            rpc_client_list[p].close()
            np.testing.assert_equal(rpc_client_list[p].is_trans_built, False)


def test_mpilauncher():
    os.environ["HETR_SERVER_PORTS"] = "51111, 51112"
    mpilauncher = MPILauncher()
    mpilauncher.launch(2, 1)

    # Check if process has launched
    assert mpilauncher.mpirun_proc.poll() is None

    mpilauncher.close()

    # Check if process has completed
    assert mpilauncher.mpirun_proc is None


@pytest.mark.multi_device
@pytest.mark.parametrize('config', [
    {
        'dataset': 'cifar10',
        'iter_count': 1,
        'batch_size': 64,
        'device_id': ('0', '1'),
        'bprop': True,
        'batch_norm': False,
    }
])
def test_hetr_benchmark(hetr_device, config):
    pytest.skip('Possible issue only on jenkins, disable until figured out.')
    """
    Description:
        Test to ensure benchmarks are working.
        Benchmark used for test is mini_resnet
    """
    from examples.benchmarks.mini_resnet import run_resnet_benchmark
    c = config
    run_resnet_benchmark(dataset=c['dataset'],
                         num_iterations=c['iter_count'],
                         n_skip=1,
                         batch_size=c['batch_size'],
                         device_id=c['device_id'],
                         transformer_type='hetr',
                         device=hetr_device,
                         bprop=c['bprop'],
                         batch_norm=c['batch_norm'],
                         visualize=False)
