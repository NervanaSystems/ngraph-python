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
from __future__ import print_function

from functools import wraps
from operator import itemgetter
# These are indirectly used by the generated code
import numpy as np
import os

from ngraph.util.pygen import PyGen, indenting
from ngraph.util.generics import generic_method

from ngraph.op_graph.op_graph import AbsoluteOp, Add, Argmax, Argmin, \
    ContiguousOp, CosOp, Op, Divide, FloorDivide, DotLowDimension, \
    Mod, Equal, ExpOp, Greater, GreaterEqual, Less, LessEqual, \
    LogOp, Max, Maximum, Min, Minimum, Multiply, NegativeOp, NotEqual, OneHotOp, \
    ReciprocalOp, Power, AssignOp, SignOp, SinOp, SqrtOp, SquareOp, RngOp, \
    Subtract, Sum, Prod, TanhOp, TensorSizeOp, Fill, TensorDescription, \
    SetItemOp, ReductionOp
from ngraph.op_graph.convolution import ConvolutionOp, update_conv, bprop_conv
from ngraph.op_graph.pooling import PoolingOp, BpropPoolOp
from ngraph.transformers.cpu.relu import ReluOp, BpropReluOp
from ngraph.op_graph.lookuptable import LookupTableOp, update_lut
from ngraph.op_graph.batchnorm import BatchnormOp
from ngraph.op_graph.ctc import CTCOp
from ngraph.op_graph.debug import PrintOp
from ngraph.transformers.passes.passes import RequiredTensorShaping, \
    CPUTensorShaping, SimplePrune
from ngraph.transformers.passes.cpulayout import CPUTensorLayout
from ngraph.transformers.passes.cpufusion import CPUFusion, FusionPass
from ngraph.transformers.passes.mkldnnpasses import MklCreateOpDescriptors, \
    MklAddLayoutConversions, MklReorderOp
from ngraph.transformers.passes.layout import AddLayoutConversions
from ngraph.transformers.passes.nviz import VizPass

from ngraph.transformers.base import Transformer, DeviceBufferStorage, \
    DeviceBufferReference, DeviceTensor, make_transformer_factory, \
    set_transformer_factory, Computation

from ngraph.op_graph.comm_nodes import CPUQueueSendOp, CPUQueueRecvOp, \
    CPUQueueGatherSendOp, CPUQueueGatherRecvOp, CPUQueueScatterSendOp, \
    CPUQueueScatterRecvOp, CPUQueueAllReduceOp


class CPUConvEngine(object):

    @staticmethod
    def get_slices(I, F, O, conv_params):
        C, D, H, W, _ = I.tensor_description.axes.lengths
        C, T, R, S, K = F.tensor_description.axes.lengths
        K, M, P, Q, _ = O.tensor_description.axes.lengths
        pad_d, pad_h, pad_w = itemgetter(*('pad_' + s for s in ('d', 'h', 'w')))(conv_params)
        str_d, str_h, str_w = itemgetter(*('str_' + s for s in ('d', 'h', 'w')))(conv_params)
        dil_d, dil_h, dil_w = itemgetter(*('dil_' + s for s in ('d', 'h', 'w')))(conv_params)
        mSlice = [CPUConvEngine.fprop_slice(m, T, D, pad_d, str_d, dil_d) for m in range(M)]
        pSlice = [CPUConvEngine.fprop_slice(p, R, H, pad_h, str_h, dil_h) for p in range(P)]
        qSlice = [CPUConvEngine.fprop_slice(q, S, W, pad_w, str_w, dil_w) for q in range(Q)]
        dSlice = [CPUConvEngine.bprop_slice(d, T, M, pad_d, str_d, dil_d) for d in range(D)]
        hSlice = [CPUConvEngine.bprop_slice(h, R, P, pad_h, str_h, dil_h) for h in range(H)]
        wSlice = [CPUConvEngine.bprop_slice(w, S, Q, pad_w, str_w, dil_w) for w in range(W)]

        return (mSlice, pSlice, qSlice, dSlice, hSlice, wSlice)

    @staticmethod
    def fprop_slice(q, S, X, padding, stride, dilation):
        f1 = None
        qs = q * stride - padding
        for s in range(S):
            x = qs + s * dilation
            if f1 is None and x >= 0 and x < X:
                x1 = x
                f1 = s
            if x < X:
                x2 = x
                f2 = s
        if f1 is None:
            return (slice(0, 0, 1), slice(0, 0, 1), 0)
        return (slice(f1, f2 + 1), slice(x1, x2 + 1, dilation), f2 - f1 + 1)

    @staticmethod
    def bprop_slice(x, S, Q, padding, stride, dilation):
        qs = x - (dilation * (S - 1) - padding)
        f1 = None
        for s in range(S):
            q = qs + s * dilation
            if q % stride == 0:
                q //= stride
                if q >= 0 and q < Q:
                    if f1 is None:
                        f1 = s
                        x1 = q
                    f2 = s
                    x2 = q
        if f1 is None:
            return (slice(0, 0, 1), slice(0, 0, 1), 0)

        f_step = 1
        while ((f_step * dilation) % stride) != 0:
            f_step += 1
        x_step = (f_step * dilation) // stride
        return (slice(f1, f2 + 1, f_step), slice(x1, x2 + 1, x_step), 0)


class CPUPoolEngine(object):

    @staticmethod
    def get_slices(I, O, pool_params):
        C, D, H, W, _ = I.tensor_description.axes.lengths
        K, M, P, Q, N = O.tensor_description.axes.lengths

        J, T, R, S, op = itemgetter(*('J', 'T', 'R', 'S', 'op'))(pool_params)
        p_c, p_d, p_h, p_w = itemgetter(*('pad_' + s for s in ('c', 'd', 'h', 'w')))(pool_params)
        s_c, s_d, s_h, s_w = itemgetter(*('str_' + s for s in ('c', 'd', 'h', 'w')))(pool_params)

        kSlice = [CPUPoolEngine.pool_slice(k, J, C, p_c, s_c) for k in range(K)]
        mSlice = [CPUPoolEngine.pool_slice(m, T, D, p_d, s_d) for m in range(M)]
        pSlice = [CPUPoolEngine.pool_slice(p, R, H, p_h, s_h) for p in range(P)]
        qSlice = [CPUPoolEngine.pool_slice(q, S, W, p_w, s_w) for q in range(Q)]
        array_argmax = np.empty((K, M, P, Q, N), dtype=np.uint32) if op == "max" else None

        return (kSlice, mSlice, pSlice, qSlice, op, array_argmax)

    @staticmethod
    def pool_slice(q, S, X, padding, strides):
        qs = q * strides - padding
        firstI = None
        for s in range(S):
            x = qs + s
            if x >= 0 and x < X:
                if firstI is None:
                    firstI = x
                lastI = x
        return (slice(firstI, lastI + 1), lastI - firstI + 1)


class CPUComputation(Computation):
    def __init__(self, transformer, computation, **kwargs):
        super(CPUComputation, self).__init__(transformer, computation, **kwargs)
        self.pool_params = dict()
        self.pool_slices = dict()
        self.conv_params = dict()
        self.conv_slices = dict()


class CPUDeviceBufferStorage(DeviceBufferStorage):

    def __init__(self, transformer, bytes, dtype, **kwargs):
        super(CPUDeviceBufferStorage, self).__init__(transformer, bytes, dtype, **kwargs)
        self.storage = None

    def create_device_tensor(self, tensor_description):
        shape_str = "_".join((str(_) for _ in tensor_description.shape))
        return CPUDeviceTensor(self.transformer, self, tensor_description,
                               name="{}_v_{}_{}".format(self.name,
                                                        tensor_description.name,
                                                        shape_str))

    @property
    def alloc_name(self):
        """
        :return: Name for allocation method.
        """
        return "alloc_" + self.name

    @property
    def update_name(self):
        """
        :return: name for update method.
        """
        return "update_" + self.name

    @property
    def ref_str(self):
        """
        :return: name to reference variable.
        """
        return self.name

    def transform_allocate(self):
        self.transformer.init_code.append("{} = None", self.ref_str)
        self.transformer.allocate_storage_code.append("def {}():", self.alloc_name)
        with indenting(self.transformer.allocate_storage_code):
            elts = self.bytes // self.dtype.itemsize
            self.transformer.allocate_storage_code.append(
                "{}(np.empty({}, dtype=np.dtype('{}')))",
                self.update_name, elts, self.dtype.name)
            self.transformer.allocate_storage_code.endl()

        self.transformer.allocate_storage_code.append("def {}(buffer):",
                                                      self.update_name)
        with indenting(self.transformer.allocate_storage_code):
            self.transformer.allocate_storage_code.append("global {}", self.ref_str)
            self.transformer.allocate_storage_code.append("{} = buffer", self.ref_str)
            self.transform_allocate_views()
        self.transformer.allocate_storage_code.endl()

        self.transformer.allocate_code.append("{}()", self.alloc_name)


class CPUDeviceBufferReference(DeviceBufferReference):

    def __init__(self, transformer, **kwargs):
        super(CPUDeviceBufferReference, self).__init__(transformer, **kwargs)


class CPUDeviceTensor(DeviceTensor):

    def __init__(self, transformer, device_buffer, tensor_description, **kwargs):
        super(CPUDeviceTensor, self).__init__(transformer, device_buffer, tensor_description,
                                              **kwargs)
        self.__tensor = None

    @property
    def tensor(self):
        if self.__tensor is None:
            self.__tensor = self.transformer.globals.get(self.name)
        return self.__tensor

    @property
    def ref_str(self):
        """
        :return: name to reference variable.
        """
        return self.name

    def transform_allocate(self):
        tensor_description = self.tensor_description
        self.transformer.init_code.append("{} = None", self.ref_str)
        self.transformer.allocate_storage_code.append(
            """global {ref}
{ref} = np.ndarray(
    shape={shape},
    dtype=np.{dtype},
    buffer=buffer,
    offset={offset},
    strides={strides})""",
            ref=self.ref_str,
            shape=tensor_description.shape,
            dtype=tensor_description.dtype,
            offset=tensor_description.offset,
            strides=tensor_description.strides)

    def get(self, tensor):
        if tensor is None:
            return self.tensor
        tensor[:] = self.tensor

    def __getitem__(self, key):
        return self.tensor.__getitem__(key)

    def __setitem__(self, key, value):
        # Temporary hack to interoperate with neon cpu backend.
        if hasattr(value, '_tensor'):
            value = value._tensor
        self.tensor.__setitem__(key, value)


def get_tensors(f):
    def tensor(x):
        if isinstance(x, CPUDeviceTensor):
            return x.tensor
        return x

    @wraps(f)
    def helper(*args):
        return f(*(tensor(arg) for arg in args))

    return helper


class CPUCodeGenerator(PyGen):

    def __init__(self, transformer, **kwargs):
        super(CPUCodeGenerator, self).__init__(prefix="op", **kwargs)
        self.transformer = transformer

    def name(self, x):
        if isinstance(x, CPUDeviceBufferStorage):
            return x.ref_str
        if isinstance(x, CPUDeviceTensor):
            return x.ref_str
        return x

    def np_reduction_axis(self, op):
        """
        Returns numpy reduction axis of an op

        Args:
            op: instance of ReductionOp

        Returns:
            tuple of numpy reduction axis
        """
        if not isinstance(op, ReductionOp):
            raise ValueError("Op %s must be an instance of ReductionOp" % op)
        input_axes = op.args[0].axes
        reduction_axes = op.reduction_axes
        np_axis = tuple([input_axes.index(axis) for axis in reduction_axes])
        return np_axis[0] if len(np_axis) == 1 else np_axis

    @property
    def pool_params(self):
        return self.transformer.current_computation.pool_params

    @property
    def pool_slices(self):
        return self.transformer.current_computation.pool_slices

    @property
    def conv_params(self):
        return self.transformer.current_computation.conv_params

    @property
    def conv_slices(self):
        return self.transformer.current_computation.conv_slices

    @property
    def send_nodes(self):
        return self.transformer.current_computation.send_nodes

    @property
    def recv_nodes(self):
        return self.transformer.current_computation.recv_nodes

    @property
    def scatter_send_nodes(self):
        return self.transformer.current_computation.scatter_send_nodes

    @property
    def scatter_recv_nodes(self):
        return self.transformer.current_computation.scatter_recv_nodes

    @property
    def gather_send_nodes(self):
        return self.transformer.current_computation.gather_send_nodes

    @property
    def gather_recv_nodes(self):
        return self.transformer.current_computation.gather_recv_nodes

    @property
    def allreduce_nodes(self):
        return self.transformer.current_computation.allreduce_nodes

    @generic_method(Op)
    def allocate_op(self, op, *args):
        pass

    @allocate_op.on_type(ConvolutionOp)
    def allocate_op(self, op, outputs, inputs, filters):
        self.conv_params[op.name] = op.conv_params
        self.conv_slices[op.name] = \
            CPUConvEngine.get_slices(inputs, filters, outputs, op.conv_params)

    @allocate_op.on_type(PoolingOp)
    def allocate_op(self, op, arrO, arrI):
        self.pool_params[op.name] = op.pool_params
        self.pool_slices[op.name] = CPUPoolEngine.get_slices(arrI, arrO, op.pool_params)

    @allocate_op.on_type(DotLowDimension)
    def allocate_op(self, op, out, x, y):
        self.append("mkldnn.init_innerproduct_fprop('{}', out={}, x={}, y={})",
                    op.name, out, x, y)

    @generic_method(Op)
    def generate_op(self, op, *args):
        if op.is_device_op:
            raise ValueError((
                "{class_name} doesn't have a generate_op method for op: {op}. "
                "In order to fix this, add a method generate_op decorated with "
                "@generate_op.on_type({op}) to class {class_name}."
            ).format(
                class_name=self.__class__.__name__,
                op=op.__class__.__name__,
            ))

    @generate_op.on_type(AbsoluteOp)
    def generate_op(self, op, out, x):
        self.append("np.abs({}, out={})", x, out)

    @generate_op.on_type(Add)
    def generate_op(self, op, out, x, y):
        self.append("mkldnn.elementwise_add('{}', I_array1={}, I_array2={}, O_array={})",
                    op.name, x, y, out)

    @generate_op.on_type(Argmax)
    def generate_op(self, op, out, x):
        self.append("np.ndarray.argmax({}, axis={}, out={})", x, self.np_reduction_axis(op), out)

    @generate_op.on_type(Argmin)
    def generate_op(self, op, out, x):
        self.append("np.ndarray.argmin({}, axis={}, out={})", x, self.np_reduction_axis(op), out)

    @generate_op.on_type(ConvolutionOp)
    def generate_op(self, op, outputs, inputs, filters):
        self.append("mkldnn.fprop_conv('{}', self.conv_slices['{}'], I={}, F={}, O={})",
                    op.name, op.name, inputs, filters, outputs)

    @generate_op.on_type(bprop_conv)
    def generate_op(self, op, outputs, delta, filters):
        self.append("mkldnn.bprop_conv('{}', self.conv_slices['{}'], E={}, F={}, gI={})",
                    op.name, op.fprop.forwarded.name, delta, filters, outputs)

    @generate_op.on_type(update_conv)
    def generate_op(self, op, outputs, delta, inputs):
        self.append("mkldnn.update_conv('{}', self.conv_slices['{}'], I={}, E={}, U={})",
                    op.name, op.fprop.forwarded.name, inputs, delta, outputs)

    @generate_op.on_type(PoolingOp)
    def generate_op(self, op, outputs, inputs):
        self.append("mkldnn.fprop_pool('{}', self.pool_slices['{}'], arrI={}, arrO={})",
                    op.name, op.name, inputs, outputs)

    @generate_op.on_type(BpropPoolOp)
    def generate_op(self, op, outputs, delta):
        self.append("mkldnn.bprop_pool('{}', self.pool_slices['{}'], arrE={}, arrD={})",
                    op.name, op.fprop.forwarded.name, delta, outputs)

    @generate_op.on_type(LookupTableOp)
    def generate_op(self, op, outputs, lut, idx):
        self.append("fprop_lut(lut={}, idx={}, axis={}, output={})",
                    lut, idx, op.lut_axis, outputs)

    @generate_op.on_type(update_lut)
    def generatea_op(self, op, outputs, delta, idx):
        if op.update:
            self.append("update_lut(error={}, idx={}, pad_idx={}, axis={}, dW={})",
                        delta, idx, op.pad_idx, op.lut_axis, outputs)

    @generate_op.on_type(CTCOp)
    def generate_op(self, op, outputs, activations, lbls, utt_lens, lbl_lens, grads):
        self.append("ctc_cpu(acts={}, lbls={}, utt_lens={}, lbl_lens={}, grads={}, costs={})",
                    activations, lbls, utt_lens, lbl_lens, grads, outputs)

    @generate_op.on_type(RngOp)
    def generate_op(self, op, out, x):
        if op.distribution == 'uniform':
            rstr = "uniform(low={low}, high={high}".format(**op.params)
        elif op.distribution == 'normal':
            rstr = "normal(loc={loc}, scale={scale}".format(**op.params)

        self.append("{out}[()] = np.random.{rstr}, size={out}.shape)", out=out, rstr=rstr)

    @generate_op.on_type(CosOp)
    def generate_op(self, op, out, x):
        self.append("np.cos({}, out={})", x, out)

    @generate_op.on_type(ContiguousOp)
    def generate_op(self, op, out, x):
        self.append("{}[()] = {}", out, x)

    @generate_op.on_type(Divide)
    def generate_op(self, op, out, x, y):
        self.append("np.divide({}, {}, out={})", x, y, out)

    @generate_op.on_type(FloorDivide)
    def generate_op(self, op, out, x, y):
        self.append("np.floor_divide({}, {}, out={})", x, y, out)

    @generate_op.on_type(Mod)
    def generate_op(self, op, out, x, y):
        self.append("np.mod({}, {}, out={})", x, y, out)

    @generate_op.on_type(DotLowDimension)
    def generate_op(self, op, out, x, y):
        self.append("mkldnn.innerproduct_fprop('{}', {}, {}, out={})",
                    op.name, x, y, out)

    @generate_op.on_type(BatchnormOp)
    def generate_op(self, op, output, inputs, gamma, bias, epsilon, mean, variance):
        self.append("mkldnn.fprop_batchnorm('{}', inputs={}, outputs={}, gamma={},\
                     bias={}, mean={}, variance={}, epsilon={})", op.name, inputs,
                     output, gamma, bias, mean, variance, epsilon)

    @generate_op.on_type(ReluOp)
    def generate_op(self, op, outputs, inputs):
        self.append("mkldnn.fprop_relu('{}', {}, {}, {})", op.name, inputs, outputs, op.slope)

    @generate_op.on_type(BpropReluOp)
    def generate_op(self, op, outputs, delta, inputs):
        self.append("mkldnn.bprop_relu('{}', {}, {}, {}, {})", op.name, delta, outputs, inputs, op.fprop.slope)

    @generate_op.on_type(Equal)
    def generate_op(self, op, out, x, y):
        self.append("np.equal({}, {}, out={})", x, y, out)

    @generate_op.on_type(ExpOp)
    def generate_op(self, op, out, x):
        self.append("np.exp({}, out={})", x, out)

    @generate_op.on_type(Fill)
    def generate_op(self, op, out, x):
        self.append("{}.fill({})", x, op.scalar)

    @generate_op.on_type(Greater)
    def generate_op(self, op, out, x, y):
        self.append("np.greater({}, {}, out={})", x, y, out)

    @generate_op.on_type(GreaterEqual)
    def generate_op(self, op, out, x, y):
        self.append("np.greater_equal({}, {}, out={})", x, y, out)

    @generate_op.on_type(Less)
    def generate_op(self, op, out, x, y):
        self.append("np.less({}, {}, out={})", x, y, out)

    @generate_op.on_type(LessEqual)
    def generate_op(self, op, out, x, y):
        self.append("np.less_equal({}, {}, out={})", x, y, out)

    @generate_op.on_type(LogOp)
    def generate_op(self, op, out, x):
        self.append("np.log({}, out={})", x, out)

    @generate_op.on_type(Max)
    def generate_op(self, op, out, x):
        self.append("np.max({}, axis={}, out={})", x, self.np_reduction_axis(op), out)

    @generate_op.on_type(Maximum)
    def generate_op(self, op, out, x, y):
        self.append("np.maximum({}, {}, out={})", x, y, out)

    @generate_op.on_type(Min)
    def generate_op(self, op, out, x):
        self.append("np.min({}, axis={}, out={})", x, self.np_reduction_axis(op), out)

    @generate_op.on_type(Minimum)
    def generate_op(self, op, out, x, y):
        self.append("np.minimum({}, {}, out={})", x, y, out)

    @generate_op.on_type(MklReorderOp)
    def generate_op(self, op, output, input):
        #self.append("{}[...] = np.copy({})", output, input)
        self.append("mkldnn.mkl_reorder('{}', {}, {})", op.name, output, input)

    @generate_op.on_type(Multiply)
    def generate_op(self, op, out, x, y):
        self.append("np.multiply({}, {}, out={})", x, y, out)

    @generate_op.on_type(NegativeOp)
    def generate_op(self, op, out, x):
        self.append("np.negative({}, out={})", x, out)

    @generate_op.on_type(NotEqual)
    def generate_op(self, op, out, x, y):
        self.append("np.not_equal({}, {}, out={})", x, y, out)

    @generate_op.on_type(OneHotOp)
    def generate_op(self, op, out, x):
        self.append("{o}[:] = np.eye({o}.shape[0])[:, {x}.astype(np.int32)]", x=x, o=out)

    @generate_op.on_type(Power)
    def generate_op(self, op, out, x, y):
        self.append("np.power({}, {}, out={})", x, y, out)

    @generate_op.on_type(PrintOp)
    def generate_op(self, op, out, x):
        if op.prefix is not None:
            self.append("""print({prefix} + ':', {x})
{out}[()] = {x}""", out=out, x=x, prefix=repr(op.prefix))
        else:
            self.append("""print({x})
{out}[()] = {x}""", out=out, x=x)

    @generate_op.on_type(ReciprocalOp)
    def generate_op(self, op, out, x):
        self.append("np.reciprocal({}, out={})", x, out)

    @generate_op.on_type(AssignOp)
    def generate_op(self, op, out, tensor, value):
        self.append("{}.__setitem__((), {})", tensor, value)

    @generate_op.on_type(SetItemOp)
    def generate_op(self, op, out, tensor, value):
        self.append("{}.__setitem__({}, {})", tensor, tuple(op.item), value)

    @generate_op.on_type(SignOp)
    def generate_op(self, op, out, x):
        self.append("np.sign({}, out=out)", x, out)

    @generate_op.on_type(SinOp)
    def generate_op(self, op, out, x):
        self.append("np.sin({}, out={})", x, out)

    @generate_op.on_type(SqrtOp)
    def generate_op(self, op, out, x):
        self.append("np.sqrt({}, out={})", x, out)

    @generate_op.on_type(SquareOp)
    def generate_op(self, op, out, x):
        self.append("np.square({}, out={})", x, out)

    @generate_op.on_type(Subtract)
    def generate_op(self, op, out, x, y):
        self.append("np.subtract({}, {}, out={})", x, y, out)

    @generate_op.on_type(Sum)
    def generate_op(self, op, out, x):
        self.append("np.sum({}, axis={}, out={})", x, self.np_reduction_axis(op), out)

    @generate_op.on_type(Prod)
    def generate_op(self, op, out, x):
        self.append("np.prod({}, axis={}, out={})", x, self.np_reduction_axis(op), out)

    @generate_op.on_type(TanhOp)
    def generate_op(self, op, out, x):
        self.append("np.tanh({}, out={})", x, out)

    @generate_op.on_type(TensorSizeOp)
    def generate_op(self, op, out):
        self.append("{}.fill({})", out, op.reduction_axes.size)

    @generate_op.on_type(CPUQueueSendOp)
    def generate_op(self, op, out, arg):
        send_id = len(self.send_nodes)
        self.send_nodes.append(op)
        self.append("self.queue_send({}, {})", send_id, arg)

    @generate_op.on_type(CPUQueueRecvOp)
    def generate_op(self, op, out):
        recv_id = len(self.recv_nodes)
        self.recv_nodes.append(op)
        self.append("self.recv_from_queue_send({}, out={})", recv_id, out)

    @generate_op.on_type(CPUQueueGatherSendOp)
    def generate_op(self, op, out, arg):
        gather_send_id = len(self.gather_send_nodes)
        self.gather_send_nodes.append(op)
        self.append("self.queue_gather_send({}, {})", gather_send_id, arg)

    @generate_op.on_type(CPUQueueGatherRecvOp)
    def generate_op(self, op, out):
        gather_recv_id = len(self.gather_recv_nodes)
        self.gather_recv_nodes.append(op)
        self.append("self.gather_recv_from_queue_gather_send({}, out={})", gather_recv_id, out)

    @generate_op.on_type(CPUQueueScatterSendOp)
    def generate_op(self, op, out, arg):
        scatter_send_id = len(self.scatter_send_nodes)
        self.scatter_send_nodes.append(op)
        self.append("self.queue_scatter_send({}, {})", scatter_send_id, arg)

    @generate_op.on_type(CPUQueueScatterRecvOp)
    def generate_op(self, op, out):
        scatter_recv_id = len(self.scatter_recv_nodes)
        self.scatter_recv_nodes.append(op)
        self.append("self.scatter_recv_from_queue_scatter_send({}, out={})",
                    scatter_recv_id, out)

    @generate_op.on_type(CPUQueueAllReduceOp)
    def generate_op(self, op, out, arg):
        allreduce_id = len(self.allreduce_nodes)
        self.allreduce_nodes.append(op)
        self.append("{}[...] = self.queue_allreduce({}, {})", out, allreduce_id, arg)
    
    @generate_op.on_type(ReductionOp)
    def generate_op(self, op, out, *args):
        # TODO(jbobba): Added to get a UT to pass. 
        # Need to look into why we need this
        pass


class CPUTransformer(Transformer):
    """
    Transformer for executing graphs on a CPU, backed by numpy.

    Given a list of ops you want to compute the results of, this transformer
    will compile the graph required to compute those results and exposes an
    evaluate method to execute the compiled graph.
    """

    transformer_name = "cpu"
    default_rtol = 1e-05
    default_atol = 1e-08

    def __init__(self, **kwargs):
        super(CPUTransformer, self).__init__(**kwargs)
        self.current_computation = None
        self.conv_engine = CPUConvEngine()
        self.init_code = CPUCodeGenerator(self)
        self.allocate_storage_code = CPUCodeGenerator(self)
        self.allocate_code = CPUCodeGenerator(self)
        self.compute_code = CPUCodeGenerator(self)
        self.code = CPUCodeGenerator(self)
        self.globals = self.code.globals
        self.n_computations = 0
        self.use_pinned_mem = False
        self.rng_seed = None
        self.initialize_mkldnn()
        add_layout_conversion = AddLayoutConversions(None)
        self.graph_passes = [CPUFusion(),
                             FusionPass(),
                             CPUTensorLayout(),
                             SimplePrune(),
                             RequiredTensorShaping(),
                             CPUTensorShaping(),
                             MklCreateOpDescriptors(self.mkldnn),
                             MklAddLayoutConversions(self.mkldnn, add_layout_conversion)
                             #,VizPass(show_axes=True,view=False)
                             ]

    def device_buffer_storage(self, bytes, dtype, name):
        """
        Make a DeviceBuffer.

        Arguments:
            bytes: Size of buffer.
            alignment: Alignment of buffer.

        Returns: A DeviceBuffer.
        """
        return CPUDeviceBufferStorage(self, bytes, dtype, name="a_" + name)

    def device_buffer_reference(self):
        """
        Make a DeviceBufferReference.

        Returns: A DeviceBufferReference.
        """
        return CPUDeviceBufferReference(self)

    def initialize_mkldnn(self):
        self.code.execute("""
from ngraph.transformers.cpu.cpuengine import Mkldnn
""")
        mkldnn_path = os.path.join(os.path.dirname(__file__), "..", "..")
        mkldnn_engine_path = os.path.join(mkldnn_path, 'mkldnn_engine.so')
        self.code.execute("mkldnn = Mkldnn('{}')".format(mkldnn_engine_path))
        self.code.execute("mkldnn.open()")
        self.mkldnn = self.globals.get("mkldnn", None)

    def start_transform_allocate(self):
        self.code.execute("""import os
import numpy as np
import ctypes as ct
import numpy.ctypeslib as npct
import itertools as itt
from ngraph.op_graph import axes
from ngraph.transformers.cpu.cpuengine import fprop_lut, update_lut
from ngraph.transformers.cpu.cpuengine import Mkldnn
from ngraph.transformers.cpu.cpuengine import ConvLocals
from ngraph.transformers.cpu.hetr import HetrLocals
from ngraph.transformers.cpu.ctc import ctc_cpu
""")

    def transform_allocate_ops(self, all_ops):
        def tensor_description_value(x):
            if isinstance(x, TensorDescription):
                return self.get_tensor_description_tensor_view(x)
            return x

        for op in all_ops:
            out = tensor_description_value(op.forwarded.tensor_description())
            call_info = (tensor_description_value(_) for _ in op.call_info())
            self.compute_code.allocate_op(op, out, *call_info)

    def finish_transform_allocate(self):
        pass

    def transform_ordered_ops(self, computation, ordered_ops, name):
        self.current_computation = computation
        if name is None:
            name = "C_" + str(self.n_computations)
        self.n_computations += 1
        self.compute_code.append("class {}(HetrLocals, ConvLocals):", name)
        with indenting(self.compute_code):
            self.compute_code.append("def __init__(self, **kwargs):")
            with indenting(self.compute_code):
                self.compute_code.append('super({}, self).__init__(**kwargs)', name)
                self.transform_allocate_ops(ordered_ops)

            self.compute_code.endl()

            self.compute_code.append("def __call__(self):")
            code_length = self.compute_code.code_length

            def tensor_description_value(x):
                if isinstance(x, TensorDescription):
                    return self.get_tensor_description_tensor_view(x)
                return x

            with indenting(self.compute_code):
                for op in ordered_ops:
                    out = tensor_description_value(op.forwarded.tensor_description())
                    call_info = (tensor_description_value(_) for _ in op.call_info())
                    self.compute_code.generate_op(op, out, *call_info)
                if code_length == self.compute_code.code_length:
                    self.compute_code.append("pass")
            self.compute_code.endl()
        self.name = name
        return name

    def finish_transform(self):
        self.code.append(self.init_code.code)
        self.code.endl()
        self.code.endl()

        self.code.append(self.allocate_storage_code.code)
        self.code.endl(2)
        self.code.append(self.allocate_code.code)
        self.code.endl(2)
        self.code.append(self.compute_code.code)
        self.code.endl()

        # import os
        # pid = os.getpid()
        # with open("code_{}{}.py".format(self.name, pid), "w") as f:
        #    f.write(self.code.code)
        # print(self.code.code)
        self.globals = self.code.compile()

        for computation in self.computations:
            cls = self.globals[computation.name]
            executor = cls(conv_params=computation.conv_params,
                           pool_params=computation.pool_params,
                           conv_slices=computation.conv_slices,
                           pool_slices=computation.pool_slices,
                           send_nodes=computation.send_nodes,
                           recv_nodes=computation.recv_nodes,
                           scatter_send_nodes=computation.scatter_send_nodes,
                           scatter_recv_nodes=computation.scatter_recv_nodes,
                           gather_send_nodes=computation.gather_send_nodes,
                           gather_recv_nodes=computation.gather_recv_nodes,
                           allreduce_nodes=computation.allreduce_nodes)
            computation.executor = executor

    def allocate_storage(self):
        pass

    def close(self):
        if self.code is not None:
            try:
                if self.code.globals.get('mkldnn', None) is not None:
                    self.code.execute('mkldnn.close()')
            except TypeError:
                pass
        self.code = None

    def consume(self, buf_index, hostlist, devlist):
        '''
        This is currently used for Aeon dataloading -- need to set things up to do actual
        device buffer allocation
        '''
        assert 0 <= buf_index < 2, 'Can only double buffer'
        hb = np.rollaxis(hostlist[buf_index], 0, hostlist[buf_index].ndim)
        if devlist[buf_index] is None:
            devlist[buf_index] = np.empty_like(hb)
        devlist[buf_index][:] = hb

    def make_computation(self, computation):
        return CPUComputation(self, computation)


set_transformer_factory(
    make_transformer_factory(CPUTransformer.transformer_name))
