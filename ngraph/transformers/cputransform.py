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

from ngraph.util.pygen import PyModule, PyGen, indenting
from ngraph.util.generics import generic_method

from ngraph.op_graph.op_graph import AbsoluteOp, Add, Argmax, Argmin, \
    ContiguousOp, CosOp, Op, Divide, FloorDivide, DotLowDimension, \
    Mod, Equal, ExpOp, Greater, GreaterEqual, Less, LessEqual, \
    LogOp, Max, Maximum, Min, Minimum, Multiply, NegativeOp, NotEqual, OneHotOp, \
    ReciprocalOp, Power, AssignOp, SignOp, SinOp, SqrtOp, SquareOp, RngOp, \
    Subtract, Sum, Prod, TanhOp, TensorSizeOp, Fill, TensorDescription, \
    ReductionOp, WriteOp, ReadOp
from ngraph.op_graph.convolution import ConvolutionOp, update_conv, bprop_conv, \
    DeconvolutionOp, DeconvDerivOp
from ngraph.op_graph.pooling import PoolingOp, BpropPoolOp
from ngraph.op_graph.lookuptable import LookupTableOp, update_lut
from ngraph.op_graph.ctc import CTCOp
from ngraph.op_graph.debug import PrintOp
from ngraph.transformers.cpu.batchnorm import BatchnormOp, BpropBatchnormOp
from ngraph.transformers.cpu.relu import ReluOp, BpropReluOp
from ngraph.transformers.passes.passes import RequiredTensorShaping, \
    CPUTensorShaping, SimplePrune
from ngraph.transformers.passes.cpulayout import CPUTensorLayout
from ngraph.transformers.passes.cpufusion import CPUFusion
from ngraph.transformers.passes.mkldnnpasses import MklCreateOpDescriptors, \
    MklAddLayoutConversions, MklReorderOp
from ngraph.transformers.passes.layout import AddLayoutConversions
from ngraph.transformers.passes.expass import SSAConversion, IndexElision, DeadCodeEliminationPass
from ngraph.transformers.passes.memlayout import MemLayoutPass
from ngraph.transformers.passes.memoptimize import MemOptimizePass
from ngraph.transformers.passes.liveness import LivenessPass

from ngraph.transformers.base import make_transformer_factory, \
    set_transformer_factory
from ngraph.transformers.extransform import ExecutionGraphTransformer, \
    DeviceTensor, DeviceTensorView, DeviceComputation

from ngraph.transformers.exop import InputDecl, OutputDecl, LiteralScalarOp

from ngraph.op_graph.comm_nodes import CPUQueueSendOp, CPUQueueRecvOp, \
    CPUQueueGatherSendOp, CPUQueueGatherRecvOp, CPUQueueScatterSendOp, \
    CPUQueueScatterRecvOp, CPUQueueAllReduceOp, CPUQueueBroadcastSendOp, \
    CPUQueueBroadcastRecvOp

from ngraph.util.trace_events import is_tracing_enabled


def align_ndarray(element_count, alignment, dtype):
    x = np.empty(element_count + (alignment - 1), dtype)
    offset = (x.ctypes.data % alignment) // dtype.itemsize
    padding = 0 if offset == 0 else (alignment - offset)
    return x[padding:padding + element_count]


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


class CPUDeviceComputation(DeviceComputation):
    def __init__(self, transformer, computation_op, **kwargs):
        super(CPUDeviceComputation, self).__init__(transformer, computation_op, **kwargs)
        self.pool_params = dict()
        self.pool_slices = dict()
        self.conv_params = dict()
        self.conv_slices = dict()


class CPUDeviceTensor(DeviceTensor):
    """
    This is the device tensor.
    """

    def __init__(self, transformer, device_computation, tensor, **kwargs):
        super(CPUDeviceTensor, self).__init__(transformer, device_computation, tensor, **kwargs)

    def make_device_tensor_view(self, tensor_view_decl):
        return CPUDeviceTensorView(self, tensor_view_decl)

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

    def codegen(self):
        start = self.buffer_pool_offset // 4
        end = start + self.size // 4
        pool_name = self.device_computation.computation_op.name
        pool_name += '_persistent_pool' if self.is_persistent else '_temporary_pool'
        dtype = self.element_type.dtype
        self.transformer.exop_codegen_tensor.append("\n# tensor size={}, offset={}",
                                                    self.size,
                                                    self.buffer_pool_offset)
        self.transformer.exop_codegen_tensor.append("{} = {}[{}:{}].view('{}')",
                                                    self.name, pool_name, start, end, dtype)

    def transform_allocate(self):
        self.transformer.init_code.append("{} = None", self.ref_str)
        self.transformer.allocate_storage_code.append("def {}():", self.alloc_name)
        with indenting(self.transformer.allocate_storage_code):
            elts = self.bytes // self.dtype.itemsize
            if self.dtype.name == 'float32':
                c_type_name = 'c_float'
            elif self.dtype.name == 'float64':
                c_type_name = 'c_double'
            else:
                c_type_name = None

            if c_type_name is not None and self.transformer.use_mlsl:
                self.transformer.allocate_storage_code.append(
                    """try:
    type_size = ctypes.sizeof(ctypes.{3}(1))
    mlsl_buf_{0} = mlsl_obj.alloc({1} * type_size, 64)
    array_{0} = ctypes.cast(mlsl_buf_{0}, ctypes.POINTER(ctypes.{3} * {1}))
    np_array_{0} = np.frombuffer(array_{0}.contents, dtype=np.dtype('{2}'))
    {0}(np_array_{0})
except NameError as error:
    print str(error)
    {0}(np.empty({1}, dtype=np.dtype('{2}')))""",
                    self.update_name, elts, self.dtype.name, c_type_name)
            else:
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


class CPUDeviceTensorView(DeviceTensorView):
    """
    This is a TensorView.
    """

    def __init__(self, device_tensor, tensor_view_decl, **kwargs):
        super(CPUDeviceTensorView, self).__init__(device_tensor, tensor_view_decl, **kwargs)
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

    def codegen(self):
        self.transformer.exop_codegen_tensor_view.append("""\n{ref} = np.ndarray(
    shape={shape},
    dtype=np.{dtype},
    buffer={buffer},
    offset={offset},
    strides={strides})""",
                                                         ref=self.ref_str,
                                                         shape=self.tensor_description.shape,
                                                         dtype=self.tensor_description.dtype,
                                                         buffer=self.device_buffer.ref_str,
                                                         offset=self.tensor_description.offset,
                                                         strides=self.tensor_description.strides)

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
            return x.tensor_decl
        return x

    @wraps(f)
    def helper(*args):
        return f(*(tensor(arg) for arg in args))

    return helper


class CPUCodeGenerator(PyGen):

    def __init__(self, transformer, **kwargs):
        super(CPUCodeGenerator, self).__init__(**kwargs)
        self.transformer = transformer

    @generic_method()
    def name(self, x):
        return x

    @name.on_type(CPUDeviceTensor)
    def name(self, x):
        return x.ref_str

    @name.on_type(CPUDeviceTensorView)
    def name(self, x):
        return x.ref_str

    @name.on_type(InputDecl)
    def name(self, x):
        return self.transformer.device_tensor_view(x.tensor_view_decl).ref_str

    @name.on_type(OutputDecl)
    def name(self, x):
        return self.transformer.device_tensor_view(x.tensor_view_decl).ref_str

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
        # TODO: Nishant (figure a better way to deal with reduction axes after fusion)
        # The reduction axis for the Max Op is Axis_0. The fusion introduces Axis_NG_SHADOW.
        # So when it tries to index on Axis_0 is gives an exception. This is a hack.
        # Thats why the TODO.
        try:
            np_axis = tuple([input_axes.index(axis) for axis in reduction_axes])
        except ValueError:
            np_axis = tuple([0, ])
        return np_axis[0] if len(np_axis) == 1 else np_axis

    @property
    def pool_params(self):
        return self.transformer.device_computation.pool_params

    @property
    def pool_slices(self):
        return self.transformer.device_computation.pool_slices

    @property
    def conv_params(self):
        return self.transformer.device_computation.conv_params

    @property
    def conv_slices(self):
        return self.transformer.device_computation.conv_slices

    @property
    def send_nodes(self):
        return self.transformer.device_computation.send_nodes

    @property
    def recv_nodes(self):
        return self.transformer.device_computation.recv_nodes

    @property
    def scatter_send_nodes(self):
        return self.transformer.device_computation.scatter_send_nodes

    @property
    def scatter_recv_nodes(self):
        return self.transformer.device_computation.scatter_recv_nodes

    @property
    def gather_send_nodes(self):
        return self.transformer.device_computation.gather_send_nodes

    @property
    def gather_recv_nodes(self):
        return self.transformer.device_computation.gather_recv_nodes

    @property
    def allreduce_nodes(self):
        return self.transformer.device_computation.allreduce_nodes

    @property
    def broadcast_send_nodes(self):
        return self.transformer.device_computation.broadcast_send_nodes

    @property
    def broadcast_recv_nodes(self):
        return self.transformer.device_computation.broadcast_recv_nodes

    @generic_method(Op)
    def allocate_op(self, op, *args):
        pass

    @allocate_op.on_type(ConvolutionOp)
    def allocate_op(self, op, outputs, inputs, filters, bias=None):
        self.conv_params[op.safe_name] = op.conv_params
        self.conv_slices[op.safe_name] = \
            CPUConvEngine.get_slices(inputs, filters, outputs, op.conv_params)

    @allocate_op.on_type(DeconvolutionOp)
    def allocate_op(self, op, outputs, inputs, filters):
        # get_slices args: Swap outputs and inputs
        self.conv_params[op.safe_name] = op.conv_params
        self.conv_slices[op.safe_name] = \
            CPUConvEngine.get_slices(outputs, filters, inputs, op.conv_params)

    @allocate_op.on_type(PoolingOp)
    def allocate_op(self, op, arrO, arrI):
        self.pool_params[op.safe_name] = op.pool_params
        self.pool_slices[op.safe_name] = CPUPoolEngine.get_slices(arrI, arrO, op.pool_params)

    def generate_op_pre(self, op):
        # exop = self.exop
        # self.append("\n# {} pre", exop.name)
        # for input_decl in exop.input_decls:
        #     input_decl_name = 'a_'+input_decl.tensor.tensor_name
        #     self.append("#    arg {}", input_decl_name)
        # for output_decl in exop.output_decls:
        #     output_decl_name = 'a_'+output_decl.tensor.tensor_name
        #     self.append("#    output_decl {}", val_name)
        if is_tracing_enabled():
            self.append("self.__profiler_start__.append(monotonic())")

    def generate_op_post(self, op):
        # exop = self.exop
        # self.append("print('{{}}'.format('{}'))", op.name)
        # for input_decl in exop.input.decls:
        #     input_decl_name = 'a_'+input_decl.tensor.tensor_name
        #     self.append("print('   arg {} = {{}}'.format({}))", input_decl_name, arg_name)
        # for val in exop.output_decls:
        #     output_decl_name = 'a_'+val.tensor.tensor_name
        #     self.append("#    output_decl {}", output_decl_name)
        #     self.append("print('   output_decl {} = {{}}'.format({}))", \
        #            output_decl_name, output_decl_name)
        if is_tracing_enabled():
            self.append("self.__profiler_stop__.append(monotonic())")

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

    @generate_op.on_type(ReadOp)
    def generate_op(self, op, out):
        pass

    @generate_op.on_type(WriteOp)
    def generate_op(self, op, out, *args):
        write_args = self.exop.write_args
        for dest, source in zip(write_args, args):
            if isinstance(source.source_output_decl.exop.op, LiteralScalarOp):
                self.append("{}[...] = {}", dest, source.source_output_decl.exop.op.scalar)
            else:
                self.append("{}[...] = {}", dest, source)

    @generate_op.on_type(AbsoluteOp)
    def generate_op(self, op, out, x):
        self.append("np.abs({}, out={})", x, out)

    @generate_op.on_type(Add)
    def generate_op(self, op, out, x, y):
        self.append("mkldnn.elementwise_add('{}', I_array1={}, I_array2={}, O_array={})",
                    op.safe_name, x, y, out)

    @generate_op.on_type(Argmax)
    def generate_op(self, op, out, x):
        self.append("np.ndarray.argmax({}, axis={}, out={})", x, self.np_reduction_axis(op), out)

    @generate_op.on_type(Argmin)
    def generate_op(self, op, out, x):
        self.append("np.ndarray.argmin({}, axis={}, out={})", x, self.np_reduction_axis(op), out)

    @generate_op.on_type(ConvolutionOp)
    def generate_op(self, op, outputs, inputs, filters, bias=None):
        self.append("mkldnn.fprop_conv('{}', self.conv_slices['{}'], I={}, F={}, B={}, O={})",
                    op.safe_name, op.safe_name, inputs, filters, bias, outputs)

    @generate_op.on_type(bprop_conv)
    def generate_op(self, op, outputs, delta, filters):
        self.append("mkldnn.bprop_conv('{}', self.conv_slices['{}'], E={}, F={}, gI={})",
                    op.safe_name, op.fprop.forwarded.safe_name, delta, filters, outputs)

    @generate_op.on_type(update_conv)
    def generate_op(self, op, outputs, delta, inputs):
        self.append("mkldnn.update_conv('{}', self.conv_slices['{}'], I={}, E={}, U={})",
                    op.safe_name, op.fprop.forwarded.safe_name, inputs, delta, outputs)

    @generate_op.on_type(DeconvolutionOp)
    def generate_op(self, op, outputs, inputs, filters):
        self.append("mkldnn.bprop_conv('{}', self.conv_slices['{}'], E={}, F={}, gI={})",
                    op.safe_name, op.safe_name, inputs, filters, outputs)

    @generate_op.on_type(DeconvDerivOp)
    def generate_op(self, op, outputs, delta, filters):
        self.append("mkldnn.fprop_conv('{}', self.conv_slices['{}'], I={}, F={}, B={},  O={})",
                    op.safe_name, op.fprop.forwarded.safe_name, delta, filters, None, outputs)

    @generate_op.on_type(PoolingOp)
    def generate_op(self, op, outputs, inputs):
        self.append("mkldnn.fprop_pool('{}', self.pool_slices['{}'], arrI={}, arrO={})",
                    op.safe_name, op.safe_name, inputs, outputs)

    @generate_op.on_type(BpropPoolOp)
    def generate_op(self, op, outputs, delta):
        self.append("mkldnn.bprop_pool('{}', self.pool_slices['{}'], arrE={}, arrD={})",
                    op.safe_name, op.fprop.forwarded.safe_name, delta, outputs)

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
        # self.append("{}[()] = {}", out, x)
        self.append("mkldnn.mkl_contiguous('{}', {}, {})",
                    op.safe_name, out, x)

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
    def generate_op(self, op, out, x, y, bias=None):
        self.append("mkldnn.innerproduct_fprop('{}', {}, {}, {}, out={})",
                    op.safe_name, x, y, bias, out)

    @generate_op.on_type(BatchnormOp)
    def generate_op(self, op, output, inputs, gamma, bias, epsilon, mean, variance):
        self.append("mkldnn.fprop_batchnorm('{}', inputs={}, outputs={}, gamma={},\
                    bias={}, mean={}, variance={}, epsilon={})", op.safe_name, inputs,
                    output, gamma, bias, mean, variance, epsilon)

    @generate_op.on_type(BpropBatchnormOp)
    def generate_op(self, op, output, delta, inputs, gamma, bias, mean, variance):
        self.append("mkldnn.bprop_batchnorm('{}', outputs={}, delta={}, inputs={}, \
                    gamma={}, bias={}, mean={}, variance={}, epsilon={})", op.safe_name, output,
                    delta, inputs, gamma, bias, mean, variance, op.fprop.eps)

    @generate_op.on_type(ReluOp)
    def generate_op(self, op, outputs, inputs):
        self.append("mkldnn.fprop_relu('{}', {}, {}, {})", op.safe_name, inputs, outputs, op.slope)

    @generate_op.on_type(BpropReluOp)
    def generate_op(self, op, outputs, delta, inputs):
        self.append("mkldnn.bprop_relu('{}', {}, {}, {}, {})",
                    op.safe_name, delta, outputs, inputs, op.fprop.slope)

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
        self.append("mkldnn.mkl_reorder('{}', {}, {})", op.safe_name, output, input)

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

    @generate_op.on_type(SignOp)
    def generate_op(self, op, out, x):
        self.append("np.sign({}, out={})", x, out)

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
    def generate_op(self, op, out, x):
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

    @generate_op.on_type(CPUQueueBroadcastSendOp)
    def generate_op(self, op, out, arg):
        broadcast_send_id = len(self.broadcast_send_nodes)
        self.broadcast_send_nodes.append(op)
        self.append("self.queue_broadcast_send({}, {})", broadcast_send_id, arg)

    @generate_op.on_type(CPUQueueBroadcastRecvOp)
    def generate_op(self, op, out):
        broadcast_recv_id = len(self.broadcast_recv_nodes)
        self.broadcast_recv_nodes.append(op)
        self.append("self.broadcast_recv_from_queue_broadcast_send({}, out={})",
                    broadcast_recv_id, out)


class CPUTransformer(ExecutionGraphTransformer):
    """
    Transformer for executing graphs on a CPU, backed by numpy.

    Given a list of ops you want to compute the results of, this transformer
    will compile the graph required to compute those results and exposes an
    evaluate method to execute the compiled graph.
    """

    transformer_name = "cpu"
    default_rtol = 1e-05
    default_atol = 1e-08

    import imp
    try:
        imp.find_module('mlsl')
        use_mlsl = True
    except ImportError:
        use_mlsl = False

    def __init__(self, **kwargs):
        super(CPUTransformer, self).__init__(**kwargs)
        self.device_computation = None
        self.conv_engine = CPUConvEngine()
        self.init_code = CPUCodeGenerator(self)
        self.allocate_storage_code = CPUCodeGenerator(self)
        self.allocate_code = CPUCodeGenerator(self)
        self.code = CPUCodeGenerator(self)
        self.globals = PyModule(prefix="op")
        self.initialize_module(self.globals)
        self.n_computations = 0
        self.use_pinned_mem = False
        self.rng_seed = None

        self.exop_codegen_pools = CPUCodeGenerator(self)
        self.exop_codegen_tensor = CPUCodeGenerator(self)
        self.exop_codegen_tensor_view = CPUCodeGenerator(self)
        self.exop_codegen = CPUCodeGenerator(self)
        self.exop_codegen_define_length = 0
        self.prefix = ''

        # from ngraph.transformers.passes.exnviz import ExVizPass
        # from ngraph.transformers.passes.verify import VerifyPass
        # from ngraph.transformers.passes.visualizemem import VisualizeMemPass
        # from ngraph.transformers.passes.dumpgraphpass import DumpGraphPass

        self.graph_passes = []
        if self.mkldnn.enabled:
            self.graph_passes.append(CPUFusion())
            self.byte_alignment = 64
        self.graph_passes += [
            # ExVizPass(view=True, filename="initial"),
            CPUTensorLayout(),
            SimplePrune(),
            RequiredTensorShaping(),
            CPUTensorShaping(),
            DeadCodeEliminationPass(),
        ]

        add_layout_conversion = AddLayoutConversions(None)
        if self.mkldnn.enabled:
            self.graph_passes.append(MklCreateOpDescriptors(mkldnn=self.mkldnn))
            self.graph_passes.append(MklAddLayoutConversions(mkldnn=self.mkldnn,
                                                             layoutpass=add_layout_conversion))

        self.graph_passes += [
            SSAConversion(),
            IndexElision(),
            # DCE here eliminates return values. Need to figure out why.
            # DeadCodeEliminationPass(),
            LivenessPass(),
            MemOptimizePass(),
            LivenessPass(),
            MemLayoutPass()
        ]
        # from ngraph.transformers.passes.dumpgraphpass import DumpGraphPass
        # self.graph_passes += [DumpGraphPass()]

        # from ngraph.transformers.passes.visualizemem import VisualizeMemPass
        # self.graph_passes += [VisualizeMemPass()]

    def finish_allocate_computation(self, computation):
        self.exop_codegen.endl(2)

    def start_define_computation(self, computation_decl):
        self.exop_codegen.append("class {}(HetrLocals, ConvLocals):",
                                 computation_decl.computation_op.name)
        with indenting(self.exop_codegen):
            self.exop_codegen.append("def __init__(self, **kwargs):")
            with indenting(self.exop_codegen):
                if is_tracing_enabled():
                    self.exop_codegen.append("""
self.__profiler_start__ = list()
self.__profiler_stop__  = list()
""")
                self.exop_codegen.append('super({}, self).__init__(**kwargs)',
                                         computation_decl.computation_op.name)
                for exop in computation_decl.exop_block:
                    output_decl = exop.output_decls[0] if len(exop.output_decls) > 0 else None
                    # TODO better way to deal with multiple values
                    self.exop_codegen.exop = exop
                    self.exop_codegen.allocate_op(exop.op, output_decl, *exop.input_decls)

            self.exop_codegen.endl()

        self.exop_codegen.indent(1)
        self.exop_codegen.append("def __call__(self):")
        self.exop_codegen.indent(1)
        self.codegen_define_length = self.exop_codegen.code_length

    def generate_exop(self, exop):
        value = exop.output_decls[0] if len(exop.output_decls) > 0 else None
        # TODO better way to deal with multiple values
        self.exop_codegen.exop = exop
        self.exop_codegen.generate_op_pre(exop.op)
        self.exop_codegen.generate_op(exop.op, value, *exop.input_decls)
        self.exop_codegen.generate_op_post(exop.op)

    def finish_define_computation(self, computation_decl):
        if self.codegen_define_length == self.exop_codegen.code_length:
            self.exop_codegen.append('pass')
        self.exop_codegen.indent(-2)

    def finish_load_computation(self, computation_decl):
        device_computation = computation_decl.device_computation
        byte_alignment = computation_decl.execution_graph.execution_state \
            .transformer.byte_alignment
        self.exop_codegen_pools.append(
            "{}_temporary_pool = align_ndarray({}, {}, np.dtype('{}'))",
            computation_decl.computation_op.name, computation_decl.temporary_max_allocated,
            byte_alignment,
            'float32')
        self.exop_codegen_pools.append(
            "{}_persistent_pool = align_ndarray({}, {}, np.dtype('{}'))",
            computation_decl.computation_op.name, computation_decl.persistent_max_allocated,
            byte_alignment,
            'float32')

        code = '#---------------------------------------------\n'
        code += '# memory pool\n'
        code += '#---------------------------------------------\n'
        code += self.exop_codegen_pools.take_code()
        code += '\n\n#---------------------------------------------\n'
        code += '# tensor\n'
        code += '#---------------------------------------------\n'
        code += self.exop_codegen_tensor.take_code()
        code += '\n\n#---------------------------------------------\n'
        code += '# tensor view\n'
        code += '#---------------------------------------------\n'
        code += self.exop_codegen_tensor_view.take_code()
        code += '\n\n#---------------------------------------------\n'
        code += '# code\n'
        code += '#---------------------------------------------\n'
        code += self.exop_codegen.take_code()
        self.globals.compile(code)
        cls = self.globals[computation_decl.computation_op.name]
        executor = cls(conv_params=device_computation.conv_params,
                       pool_params=device_computation.pool_params,
                       conv_slices=device_computation.conv_slices,
                       pool_slices=device_computation.pool_slices,
                       send_nodes=device_computation.send_nodes,
                       recv_nodes=device_computation.recv_nodes,
                       scatter_send_nodes=device_computation.scatter_send_nodes,
                       scatter_recv_nodes=device_computation.scatter_recv_nodes,
                       gather_send_nodes=device_computation.gather_send_nodes,
                       gather_recv_nodes=device_computation.gather_recv_nodes,
                       allreduce_nodes=device_computation.allreduce_nodes,
                       broadcast_send_nodes=device_computation.broadcast_send_nodes,
                       broadcast_recv_nodes=device_computation.broadcast_recv_nodes)
        return executor

    def make_device_tensor(self, computation, tensor_decl):
        """
        Make a DeviceTensor.

        Arguments:
            computation:
            tensor_decl: A TensorDecl.

        Returns: A DeviceTensor.
        """
        return CPUDeviceTensor(self, computation, tensor_decl)

    def initialize_module(self, module):
        module.execute("""from __future__ import print_function
from builtins import print
import os
import numpy as np
import ctypes as ct
import numpy.ctypeslib as npct
import itertools as itt
from monotonic import monotonic as monotonic
try:
    import mlsl
    import ctypes
except ImportError:
    pass
from ngraph.op_graph import axes
from ngraph.transformers.cpu.cpuengine import fprop_lut, update_lut
from ngraph.transformers.cpu.cpuengine import Mkldnn
from ngraph.transformers.cpu.cpuengine import ConvLocals
from ngraph.transformers.cpu.hetr import HetrLocals
from ngraph.transformers.cpu.ctc import ctc_cpu
from ngraph.transformers.cputransform import align_ndarray
        """)

        mkldnn_path = os.path.join(os.path.dirname(__file__), "..", "..")
        mkldnn_engine_path = os.path.join(mkldnn_path, 'mkldnn_engine.so')
        module.execute("mkldnn = Mkldnn(r'{}')".format(mkldnn_engine_path))
        module.execute("mkldnn.open()")
        self.mkldnn = module['mkldnn']
        if self.use_mlsl:
            module.execute("mlsl_obj = mlsl.MLSL()")
            module.execute("mlsl_obj.init()")

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

    def allocate_storage(self):
        pass

    def close(self):
        if self.code is not None:
            try:
                if self.globals.get('mkldnn', None) is not None:
                    self.globals.execute('mkldnn.close()')
                if self.globals.get('mlsl_obj', None) is not None:
                    for device_buffer in self.device_buffers:
                        self.globals.execute("mlsl_obj.free({}.__array_interface__['data'][0])"
                                             .format(device_buffer.ref_str))

                    self.globals.execute('mlsl_obj.finalize()')
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
        return CPUDeviceComputation(self, computation)


set_transformer_factory(
    make_transformer_factory(CPUTransformer.transformer_name))
