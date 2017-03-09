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
import numpy as np  # noqa
import itertools as itt  # noqa
from ngraph.op_graph import axes  # noqa

from ngraph.util.pygen import PyGen, indenting
from ngraph.util.generics import generic_method

from ngraph.op_graph.op_graph import AbsoluteOp, Add, Argmax, Argmin, \
    ContiguousOp, CosOp, Op, Divide, DotLowDimension, \
    Mod, Equal, ExpOp, Greater, GreaterEqual, Less, LessEqual, \
    LogOp, Max, Maximum, Min, Minimum, Multiply, NegativeOp, NotEqual, OneHotOp, \
    ReciprocalOp, Power, AssignOp, SignOp, SinOp, SqrtOp, SquareOp, RngOp, \
    Subtract, Sum, Prod, TanhOp, TensorSizeOp, Fill, TensorDescription, \
    SetItemOp, ReductionOp
from ngraph.op_graph.convolution import ConvolutionOp, update_conv, bprop_conv
from ngraph.op_graph.pooling import PoolingOp, BpropPoolOp
from ngraph.op_graph.lookuptable import LookupTableOp, update_lut
from ngraph.op_graph.debug import PrintOp
from ngraph.transformers.passes.passes import RequiredTensorShaping, \
    CPUTensorShaping, SimplePrune
from ngraph.transformers.passes.cpulayout import CPUTensorLayout

from ngraph.transformers.base import Transformer, DeviceBufferStorage, \
    DeviceBufferReference, DeviceTensor, make_transformer_factory, \
    set_transformer_factory

from ngraph.factory.comm_nodes import NumpyQueueSendOp, NumpyQueueRecvOp, \
    NumpyQueueGatherSendOp, NumpyQueueGatherRecvOp, NumpyQueueScatterSendOp, \
    NumpyQueueScatterRecvOp


class NumPyConvEngine(object):

    @staticmethod
    def all_conv_code():
        pycode = """
        def fprop_conv(self, conv_slices, I, F, O):
            mSlice, pSlice, qSlice, _, _, _ = conv_slices
            K, M, P, Q, N = O.shape

            for (m, mS), (p, pS), (q, qS) in itt.product(enumerate(mSlice),
                                                         enumerate(pSlice),
                                                         enumerate(qSlice)):
                sliceT, sliceD, _ = mS
                sliceR, sliceH, _ = pS
                sliceS, sliceW, _ = qS
                slicedF = F[:, sliceT, sliceR, sliceS, :].reshape((-1, K))
                slicedI = I[:, sliceD, sliceH, sliceW, :].reshape((-1, N))
                O[:, m, p, q, :] = np.dot(slicedF.T, slicedI)

        def bprop_conv(self, conv_slices, E, F, gI):
            _, _, _, mSlice, pSlice, qSlice = conv_slices
            F = np.transpose(F[:, ::-1, ::-1, ::-1, :], (4, 1, 2, 3, 0)).copy()
            K, M, P, Q, N = gI.shape

            for (m, mS), (p, pS), (q, qS) in itt.product(enumerate(mSlice),
                                                         enumerate(pSlice),
                                                         enumerate(qSlice)):
                sliceT, sliceD, _ = mS
                sliceR, sliceH, _ = pS
                sliceS, sliceW, _ = qS
                slicedF = F[:, sliceT, sliceR, sliceS, :].reshape((-1, K))
                slicedI = E[:, sliceD, sliceH, sliceW, :].reshape((-1, N))
                gI[:, m, p, q, :] = np.dot(slicedF.T, slicedI)

        def update_conv(self, conv_slices, I, E, U):
            mSlice, pSlice, qSlice, _, _, _ = conv_slices
            K, M, P, Q, N = E.shape
            C, _, _, _, K = U.shape
            U.fill(0.0)

            for (m, mS), (p, pS), (q, qS) in itt.product(enumerate(mSlice),
                                                         enumerate(pSlice),
                                                         enumerate(qSlice)):
                sliceT, sliceD, tlen = mS
                sliceR, sliceH, rlen = pS
                sliceS, sliceW, slen = qS
                slicedI = I[:, sliceD, sliceH, sliceW, :].reshape((-1, N))
                slicedE = E[:, m, p, q, :]
                update = np.dot(slicedI, slicedE.T).reshape((C, tlen, rlen, slen, K))
                U[:, sliceT, sliceR, sliceS, :] += update

        def fprop_pool(self, pool_slices, arrI, arrO):
            kSlice, mSlice, pSlice, qSlice, op, arrA = pool_slices
            K, M, P, Q, N = arrO.shape


            for (k, kS), (m, mS), (p, pS), (q, qS) in itt.product(enumerate(kSlice),
                                                                  enumerate(mSlice),
                                                                  enumerate(pSlice),
                                                                  enumerate(qSlice)):
                sliceC, _ = kS
                sliceD, _ = mS
                sliceH, _ = pS
                sliceW, _ = qS

                sliceI = arrI[sliceC, sliceD, sliceH, sliceW, :].reshape(-1, N)
                if op == "max":
                    arrA[k, m, p, q, :] = np.argmax(sliceI, axis=0)
                    arrO[k, m, p, q, :] = np.max(sliceI, axis=0)
                elif op == "avg":
                    arrO[k, m, p, q, :] = np.mean(sliceI, axis=0)
                elif op == "l2":
                    arrO[k, m, p, q, :] = np.sqrt(np.sum(np.square(sliceI), axis=0))

        def bprop_pool(self, pool_slices, arrE, arrD):
            kSlice, mSlice, pSlice, qSlice, op, arrA = pool_slices
            arrD[:] = 0
            K, M, P, Q, N = arrE.shape

            for (k, kS), (m, mS), (p, pS), (q, qS) in itt.product(enumerate(kSlice),
                                                                  enumerate(mSlice),
                                                                  enumerate(pSlice),
                                                                  enumerate(qSlice)):
                sliceC, clen = kS
                sliceD, dlen = mS
                sliceH, hlen = pS
                sliceW, wlen = qS

                patch_in = (sliceC, sliceD, sliceH, sliceW, slice(None))
                patch_out = (k, m, p, q, slice(None))
                sliceB = arrD[patch_in].reshape((-1, N))
                if op == "max":
                    max_n = arrA[patch_out]
                    sliceB[max_n, list(range(N))] += arrE[patch_out]
                elif op == "avg":
                    sliceB += arrE[patch_out] * (1.0 / sliceB.shape[0])
                else:
                    raise NotImplementedError
                arrD[patch_in] = sliceB.reshape((clen, dlen, hlen, wlen, N))

        """
        return pycode

    @staticmethod
    def get_slices(I, F, O, conv_params):
        C, D, H, W, _ = I.tensor_description.axes.lengths
        C, T, R, S, K = F.tensor_description.axes.lengths
        K, M, P, Q, _ = O.tensor_description.axes.lengths
        pad_d, pad_h, pad_w = itemgetter(*('pad_' + s for s in ('d', 'h', 'w')))(conv_params)
        str_d, str_h, str_w = itemgetter(*('str_' + s for s in ('d', 'h', 'w')))(conv_params)
        dil_d, dil_h, dil_w = itemgetter(*('dil_' + s for s in ('d', 'h', 'w')))(conv_params)
        mSlice = [NumPyConvEngine.fprop_slice(m, T, D, pad_d, str_d, dil_d) for m in range(M)]
        pSlice = [NumPyConvEngine.fprop_slice(p, R, H, pad_h, str_h, dil_h) for p in range(P)]
        qSlice = [NumPyConvEngine.fprop_slice(q, S, W, pad_w, str_w, dil_w) for q in range(Q)]
        dSlice = [NumPyConvEngine.bprop_slice(d, T, M, pad_d, str_d, dil_d) for d in range(D)]
        hSlice = [NumPyConvEngine.bprop_slice(h, R, P, pad_h, str_h, dil_h) for h in range(H)]
        wSlice = [NumPyConvEngine.bprop_slice(w, S, Q, pad_w, str_w, dil_w) for w in range(W)]

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


class NumPyPoolEngine(object):

    @staticmethod
    def get_slices(I, O, pool_params):
        C, D, H, W, _ = I.tensor_description.axes.lengths
        K, M, P, Q, N = O.tensor_description.axes.lengths

        J, T, R, S, op = itemgetter(*('J', 'T', 'R', 'S', 'op'))(pool_params)
        p_c, p_d, p_h, p_w = itemgetter(*('pad_' + s for s in ('c', 'd', 'h', 'w')))(pool_params)
        s_c, s_d, s_h, s_w = itemgetter(*('str_' + s for s in ('c', 'd', 'h', 'w')))(pool_params)

        kSlice = [NumPyPoolEngine.pool_slice(k, J, C, p_c, s_c) for k in range(K)]
        mSlice = [NumPyPoolEngine.pool_slice(m, T, D, p_d, s_d) for m in range(M)]
        pSlice = [NumPyPoolEngine.pool_slice(p, R, H, p_h, s_h) for p in range(P)]
        qSlice = [NumPyPoolEngine.pool_slice(q, S, W, p_w, s_w) for q in range(Q)]
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


class NumPyCodeEngine(object):

    @staticmethod
    def lut_code():
        pycode = """
        def fprop_lut(self, lut, idx, axis, output):
            output[:] = lut.take(idx.astype(int), axis)

        def update_lut(self, error, idx, pad_idx, axis, dW):
            dW[:] = 0
            idx = idx.astype(int)
            unqidx, inv = np.unique(idx, return_inverse=True)
            groups = [np.where(inv == i) for i in range(len(unqidx))]
            for (wrd_id, group) in zip(unqidx, groups):
                if wrd_id != pad_idx:
                    if axis == 0:
                        dW[wrd_id, :] = np.sum(error.take(group[0], axis=axis), axis=axis)
                    else:
                        dW[:, wrd_id] = np.sum(error.take(group[0], axis=axis), axis=axis)

        """
        return pycode


class NumPyDeviceBufferStorage(DeviceBufferStorage):

    def __init__(self, transformer, bytes, dtype, **kwargs):
        super(NumPyDeviceBufferStorage, self).__init__(transformer, bytes, dtype, **kwargs)
        self.storage = None

    def create_device_tensor(self, tensor_description):
        shape_str = "_".join((str(_) for _ in tensor_description.shape))
        return NumPyDeviceTensor(self.transformer, self, tensor_description,
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
        return "self." + self.name

    def transform_allocate(self):
        self.transformer.init_code.append("{} = None", self.ref_str)
        self.transformer.allocate_storage_code.append("def {}(self):", self.alloc_name)
        with indenting(self.transformer.allocate_storage_code):
            elts = self.bytes // self.dtype.itemsize
            self.transformer.allocate_storage_code.append(
                """
                self.{}(np.empty({}, dtype=np.dtype('{}')))
                """,
                self.update_name, elts, self.dtype.name)
            self.transformer.allocate_storage_code.endl()

        self.transformer.allocate_storage_code.append("def {}(self, buffer):",
                                                      self.update_name)
        with indenting(self.transformer.allocate_storage_code):
            self.transformer.allocate_storage_code.append("{} = buffer", self.ref_str)
            self.transform_allocate_views()
        self.transformer.allocate_storage_code.endl()

        self.transformer.allocate_code.append("self.{}()", self.alloc_name)


class NumPyDeviceBufferReference(DeviceBufferReference):

    def __init__(self, transformer, **kwargs):
        super(NumPyDeviceBufferReference, self).__init__(transformer, **kwargs)


class NumPyDeviceTensor(DeviceTensor):

    def __init__(self, transformer, device_buffer, tensor_description, **kwargs):
        super(NumPyDeviceTensor, self).__init__(transformer, device_buffer, tensor_description,
                                                **kwargs)
        self.__tensor = None

    @property
    def tensor(self):
        if self.__tensor is None:
            self.__tensor = getattr(self.transformer.model, self.name)
        return self.__tensor

    @property
    def ref_str(self):
        """
        :return: name to reference variable.
        """
        return "self." + self.name

    def transform_allocate(self):
        tensor_description = self.tensor_description
        self.transformer.init_code.append("{} = None", self.ref_str)
        self.transformer.allocate_storage_code.append(
            """
            {ref} = np.ndarray(
                shape={shape},
                dtype=np.{dtype},
                buffer=buffer,
                offset={offset},
                strides={strides})
            """,
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
        if isinstance(x, NumPyDeviceTensor):
            return x.tensor
        return x

    @wraps(f)
    def helper(*args):
        return f(*(tensor(arg) for arg in args))

    return helper


class NumPyCodeGenerator(PyGen):

    def __init__(self, **kwargs):
        super(NumPyCodeGenerator, self).__init__(**kwargs)
        self.conv_params = dict()
        self.conv_slices = dict()
        self.pool_params = dict()
        self.pool_slices = dict()
        self.send_nodes = dict()
        self.recv_nodes = dict()
        self.scatter_send_nodes = dict()
        self.scatter_recv_nodes = dict()
        self.gather_send_nodes = dict()
        self.gather_recv_nodes = dict()

    def name(self, x):
        if isinstance(x, NumPyDeviceBufferStorage):
            return x.ref_str
        if isinstance(x, NumPyDeviceTensor):
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
        self.append("np.add({}, {}, out={})", x, y, out)

    @generate_op.on_type(Argmax)
    def generate_op(self, op, out, x):
        self.append("np.ndarray.argmax({}, axis={}, out={})", x, self.np_reduction_axis(op), out)

    @generate_op.on_type(Argmin)
    def generate_op(self, op, out, x):
        self.append("np.ndarray.argmin({}, axis={}, out={})", x, self.np_reduction_axis(op), out)

    @generate_op.on_type(ConvolutionOp)
    def generate_op(self, op, outputs, inputs, filters):
        self.conv_params[op.index] = op.conv_params
        self.conv_slices[op.index] = \
            NumPyConvEngine.get_slices(inputs, filters, outputs, op.conv_params)
        self.append("self.fprop_conv(self.conv_slices[{}], I={}, F={}, O={})",
                    op.index, inputs, filters, outputs)

    @generate_op.on_type(bprop_conv)
    def generate_op(self, op, outputs, delta, filters):
        self.append("self.bprop_conv(self.conv_slices[{}], E={}, F={}, gI={})",
                    op.index, delta, filters, outputs)

    @generate_op.on_type(update_conv)
    def generate_op(self, op, outputs, delta, inputs):
        self.append("self.update_conv(self.conv_slices[{}], I={}, E={}, U={})",
                    op.index, inputs, delta, outputs)

    @generate_op.on_type(PoolingOp)
    def generate_op(self, op, outputs, inputs):
        self.pool_params[op.index] = op.pool_params
        self.pool_slices[op.index] = NumPyPoolEngine.get_slices(inputs, outputs, op.pool_params)
        self.append("self.fprop_pool(self.pool_slices[{}], arrI={}, arrO={})",
                    op.index, inputs, outputs)

    @generate_op.on_type(BpropPoolOp)
    def generate_op(self, op, outputs, delta):
        self.append("self.bprop_pool(self.pool_slices[{}], arrE={}, arrD={})",
                    op.index, delta, outputs)

    @generate_op.on_type(LookupTableOp)
    def generate_op(self, op, outputs, lut, idx):
        self.append("self.fprop_lut(lut={}, idx={}, axis={}, output={})",
                    lut, idx, op.lut_axis, outputs)

    @generate_op.on_type(update_lut)
    def generatea_op(self, op, outputs, delta, idx):
        if op.update:
            self.append("self.update_lut(error={}, idx={}, pad_idx={}, axis={}, dW={})",
                        delta, idx, op.pad_idx, op.lut_axis, outputs)

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

    @generate_op.on_type(Mod)
    def generate_op(self, op, out, x, y):
        self.append("np.mod({}, {}, out={})", x, y, out)

    @generate_op.on_type(DotLowDimension)
    def generate_op(self, op, out, x, y):
        self.append("""np.dot({}, {}, out={})""", x, y, out)

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
        self.append("""
        {o}[:] = np.eye({o}.shape[0])[:, {x}.astype(np.int32)]
        """, x=x, o=out)

    @generate_op.on_type(Power)
    def generate_op(self, op, out, x, y):
        self.append("np.power({}, {}, out={})", x, y, out)

    @generate_op.on_type(PrintOp)
    def generate_op(self, op, out, x):
        if op.prefix is not None:
            self.append("""
                print({prefix} + ':', {x})
                {out}[()] = {x}
            """, out=out, x=x, prefix=repr(op.prefix))
        else:
            self.append("""
                print({x})
                {out}[()] = {x}
            """, out=out, x=x)

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

    @generate_op.on_type(NumpyQueueSendOp)
    def generate_op(self, op, out, *args):
        send_id = len(self.send_nodes)
        self.send_nodes[send_id] = op
        self.append("self.queue_send({})", send_id)

    @generate_op.on_type(NumpyQueueRecvOp)
    def generate_op(self, op, out, *args):
        recv_id = len(self.recv_nodes)
        self.recv_nodes[recv_id] = op
        self.append("{} = self.recv_from_queue_send({})", out, recv_id)

    @generate_op.on_type(NumpyQueueGatherSendOp)
    def generate_op(self, op, out, *args):
        gather_send_id = len(self.gather_send_nodes)
        self.gather_send_nodes[gather_send_id] = op
        self.append("self.queue_gather_send({})", gather_send_id)

    @generate_op.on_type(NumpyQueueGatherRecvOp)
    def generate_op(self, op, out, *args):
        gather_recv_id = len(self.gather_recv_nodes)
        self.gather_recv_nodes[gather_recv_id] = op
        self.append("{}[:] = self.gather_recv_from_queue_gather_send({})", out, gather_recv_id)

    @generate_op.on_type(NumpyQueueScatterSendOp)
    def generate_op(self, op, out, *args):
        scatter_send_id = len(self.scatter_send_nodes)
        self.scatter_send_nodes[scatter_send_id] = op
        self.append("self.queue_scatter_send({})", scatter_send_id)

    @generate_op.on_type(NumpyQueueScatterRecvOp)
    def generate_op(self, op, out, *args):
        scatter_recv_id = len(self.scatter_recv_nodes)
        self.scatter_recv_nodes[scatter_recv_id] = op
        self.append("{}[:] = self.scatter_recv_from_queue_scatter_send({})", out, scatter_recv_id)


class NumPyTransformer(Transformer):
    """
    Transformer for executing graphs on a CPU, backed by numpy.

    Given a list of ops you want to compute the results of, this transformer
    will compile the graph required to compute those results and exposes an
    evaluate method to execute the compiled graph.
    """

    transformer_name = "numpy"
    default_rtol = 1e-05
    default_atol = 1e-08

    def __init__(self, **kwargs):
        super(NumPyTransformer, self).__init__(**kwargs)
        self.conv_engine = NumPyConvEngine()
        self.init_code = NumPyCodeGenerator()
        self.allocate_storage_code = NumPyCodeGenerator()
        self.allocate_code = NumPyCodeGenerator()
        self.compute_code = NumPyCodeGenerator()
        self.code = NumPyCodeGenerator()
        self.model = None
        self.n_computations = 0
        self.use_pinned_mem = False
        self.rng_seed = None
        self.graph_passes = [CPUTensorLayout(),
                             SimplePrune(),
                             RequiredTensorShaping(),
                             CPUTensorShaping()]

    def device_buffer_storage(self, bytes, dtype, name):
        """
        Make a DeviceBuffer.

        Arguments:
            bytes: Size of buffer.
            alignment: Alignment of buffer.

        Returns: A DeviceBuffer.
        """
        return NumPyDeviceBufferStorage(self, bytes, dtype, name="a_" + name)

    def device_buffer_reference(self):
        """
        Make a DeviceBufferReference.

        Returns: A DeviceBufferReference.
        """
        return NumPyDeviceBufferReference(self)

    def start_transform_allocate(self):
        self.init_code.append("""def __init__(self):""")
        self.init_code.indent(1)
        self.allocate_code.append("""def allocate(self):""")
        self.allocate_code.indent(1)

    def finish_transform_allocate(self):
        pass

    def transform_ordered_ops(self, ordered_ops, name):
        if name is None:
            name = "c_" + str(self.n_computations)
        self.n_computations += 1
        self.compute_code.append("def {}(self):", name)
        code = self.compute_code.code

        def tensor_description_value(x):
            if isinstance(x, TensorDescription):
                return x.value
            return x

        with indenting(self.compute_code):
            for op in ordered_ops:
                out = tensor_description_value(op.tensor_description())
                call_info = (tensor_description_value(_) for _ in op.call_info())
                self.compute_code.generate_op(op, out, *call_info)
            if code is self.compute_code.code:
                self.compute_code.append("pass")
        self.compute_code.endl()
        self.name = name
        return name

    def finish_transform(self):
        if self.model is not None:
            return

        self.code.append(" class Model(object):")
        with indenting(self.code):
            if len(self.device_buffers) == 0:
                self.init_code.append("pass")
            self.code.append(self.init_code.code)
            self.code.endl()

            self.code.append(NumPyConvEngine.all_conv_code())
            self.code.append(NumPyCodeEngine.lut_code())
            self.code.endl()

            self.code.append(self.allocate_storage_code.code)
            self.code.endl()
            if len(self.device_buffers) == 0:
                self.allocate_code.append("pass")
            self.code.append(self.allocate_code.code)
            self.code.endl(2)
            self.code.append(self.compute_code.code)

            # with open("code_{}.py".format(self.name), "w") as f:
            #    f.write(self.code.code)

        r = self.code.compile("op", globals())
        self.model = r['Model']

        def queue_send(self, send_id):
            send_op = self.send_nodes[send_id]
            q = send_op.queue

            # TODO
            # below converts DeviceTensor to numpy array
            # should we instead serialize DeviceTensor?
            x_devicetensor = send_op.args[0].value
            x_nparr = x_devicetensor.get(None)
            q.put(x_nparr)

        def queue_recv(self, recv_id):
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

        def queue_gather_recv(self, gather_recv_id):
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

        def queue_scatter_recv(self, scatter_recv_id):
            scatter_recv_op = self.scatter_recv_nodes[scatter_recv_id]
            q = scatter_recv_op.shared_queues[scatter_recv_op.idx]
            x = q.get()
            return x

        self.model.recv_from_queue_send = queue_recv
        self.model.queue_send = queue_send

        self.model.gather_recv_from_queue_gather_send = queue_gather_recv
        self.model.queue_gather_send = queue_gather_send

        self.model.scatter_recv_from_queue_scatter_send = queue_scatter_recv
        self.model.queue_scatter_send = queue_scatter_send

        self.model = self.model()

        self.model.send_nodes = self.compute_code.send_nodes
        self.model.recv_nodes = self.compute_code.recv_nodes

        self.model.gather_send_nodes = self.compute_code.gather_send_nodes
        self.model.gather_recv_nodes = self.compute_code.gather_recv_nodes

        self.model.scatter_send_nodes = self.compute_code.scatter_send_nodes
        self.model.scatter_recv_nodes = self.compute_code.scatter_recv_nodes

        self.model.conv_params = self.compute_code.conv_params
        self.model.pool_params = self.compute_code.pool_params
        self.model.conv_slices = self.compute_code.conv_slices
        self.model.pool_slices = self.compute_code.pool_slices

        for computation in self.computations:
            executor = getattr(self.model, computation.name)
            computation.executor = executor

    def allocate_storage(self):
        self.model.allocate()

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


set_transformer_factory(
    make_transformer_factory(NumPyTransformer.transformer_name))
