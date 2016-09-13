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

import math
from functools import wraps

# These are indirectly used by the generated code
import numpy as np  # noqa
from neon.backends.layer_cpu import ConvLayer  # noqa
from ngraph.op_graph import arrayaxes  # noqa

from ngraph.util.pygen import PyGen, indenting
from ngraph.util.generics import generic_method

from ngraph.op_graph.axes_ops import dimshuffle
from ngraph.op_graph.convolution import convolution1d
from ngraph.op_graph.debug import PrintOp
from ngraph.op_graph.op_graph import AxesCastOp
from ngraph.op_graph.op_graph import Broadcast
from ngraph.op_graph.op_graph import absolute, add, argmax, argmin, cos, divide, dot, equal, exp, \
    greater, greater_equal, less, less_equal, log, max, maximum, min, minimum, multiply, \
    negative, not_equal, onehot, power, reciprocal, SetItem, sign, sin, sqrt, square, subtract, \
    sum, tanh, tensor_size, Fill, TensorDescription, Unslice, Stack

from ngraph.transformers.base import Transformer, DeviceBufferStorage, DeviceBufferReference, \
    DeviceTensor


class proxy_backend(object):
    """ a fake neon backend to make ConvLayer not raise an exception. """
    # TODO: refactor away

    def check_caffe_compat(self):
        """ no caffe compat for now """
        return False

    def output_dim(self, X, S, padding, strides, pooling=False):
        """
        Compute along 1 dimension, with these sizes, what will be the output dimension.

        Arguments:
            X (int): input data dimension
            S (int): filter dimension
            padding (int): padding on each side
            strides (int): striding
            pooling (bool): flag for setting pooling layer size
        """
        if X < S:
            raise ValueError((
                'filter dimension {S} can not be large than input data '
                'dimension {X}'
            ).format(S=S, X=X))

        if self.check_caffe_compat() and pooling:
            size = int(math.ceil((float(X - S + 2 * padding) / strides))) + 1
            if padding > 0 and (size - 1) * strides >= X + padding:
                # decrement size if last pooling op is completely in padding
                size -= 1
        else:
            # normal neon output size determination
            size = ((X - S + 2 * padding) // strides) + 1

        if pooling and padding >= S:
            raise ValueError("Padding dim %d incompatible with filter size %d" % (padding, S))

        return size


class proxy_tensor(object):
    """ A fake CPUTensor to make old neon implementation of ConvLayer happy """
    # TODO: refactor away

    def __init__(self, tensor):
        self._tensor = tensor


class NumPyDeviceBufferStorage(DeviceBufferStorage):
    def __init__(self, transformer, bytes, dtype, **kwargs):
        super(NumPyDeviceBufferStorage, self).__init__(transformer, bytes, dtype, **kwargs)
        self.storage = None

    def create_device_tensor(self, tensor_description):
        shape_str = "_".join((str(_) for _ in tensor_description.shape))
        return NumPyDeviceTensor(self.transformer, self, tensor_description,
                                 name="v_" + tensor_description.name + "_" + shape_str)

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
        self.tensor.__setitem__(key, value)

    def reshape(self, shape):
        """Temporary for conv"""
        # TODO Remove when CONV is finished
        return self.tensor.reshape(shape)


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

    def name(self, x):
        if isinstance(x, NumPyDeviceBufferStorage):
            return x.ref_str
        if isinstance(x, NumPyDeviceTensor):
            return x.ref_str
        return x

    @generic_method
    def generate_op(self, op, *args):
        if op.device_op:
            raise ValueError((
                "{class_name} doesn't have a generate_op method for op: {op}. "
                "In order to fix this, add a method generate_op decorated with "
                "@generate_op.on_type({op}) to class {class_name}."
            ).format(
                class_name=self.__class__.__name__,
                op=op.__class__.__name__,
            ))

    @generate_op.on_type(absolute)
    def generate_op(self, op, out, x):
        self.append("np.abs({}, out={}", x, out)

    @generate_op.on_type(add)
    def generate_op(self, op, out, x, y):
        self.append("np.add({}, {}, out={})", x, y, out)

    @generate_op.on_type(argmax)
    def generate_op(self, op, out, x):
        self.append("np.ndarray.argmax({}, 0, out={})", x, out)

    @generate_op.on_type(argmin)
    def generate_op(self, op, out, x):
        self.append("np.ndarray.argmin({}, 0, out={})", x, out)

    @generate_op.on_type(Broadcast)
    def generate_op(self, op, out, x):
        pass

    @generate_op.on_type(AxesCastOp)
    def generate_op(self, op, out, x):
        pass

    @generate_op.on_type(convolution1d)
    def generate_op(self, op, output, input, filter):
        self.append(
            """
            input = {input}
            input = input.reshape((
                # from: channels, width, batch_size
                # to: channels, depth, height, width, batch_size
                input.shape[0], 1, 1, input.shape[1], input.shape[2]
            ))

            filter = {filter}

            filter = filter.reshape((
                # from: channels_in, width, channels_out
                # to: channels_in, depth, height, width, channels_out
                filter.shape[0], 1, 1, filter.shape[1], filter.shape[2]
            ))

            neon_conv_layer = ConvLayer(
                proxy_backend(), {output}.dtype,
                N={bsz},
                C={input_shape}[0],
                D=1,
                H=1,
                W={input_shape}[1],

                K={filter_shape}[2],
                T=1,
                R=1,
                S={filter_shape}[1],

                pad_d=0, pad_h=0, pad_w=0,
                str_d=1, str_h=1, str_w=1,
            )

            # neon_conv_layer...
            neon_conv_layer.xprop_conv(
                proxy_tensor(input),
                proxy_tensor(filter),
                proxy_tensor({output}),
            )
            """,
            output=output,
            input=input,
            filter=filter,
            input_shape=[a.length for a in op._input_shape],
            filter_shape=[a.length for a in op._filter_shape],
            bsz=op.batch_axis.length
        )

    @generate_op.on_type(cos)
    def generate_op(self, op, out, x):
        self.append("np.cos({}, out={})", x, out)

    @generate_op.on_type(divide)
    def generate_op(self, op, out, x, y):
        self.append("np.divide({}, {}, out={})", x, y, out)

    @generate_op.on_type(dot)
    def generate_op(self, op, out, o, x, y):
        self.append("np.dot({}, {}, out={})", x, y, o)

    @generate_op.on_type(equal)
    def generate_op(self, op, out, x, y):
        self.append("np.equal({}, {}, out={})", x, y, out)

    @generate_op.on_type(exp)
    def generate_op(self, op, out, x):
        self.append("np.exp({}, out={})", x, out)

    @generate_op.on_type(Fill)
    def generate_op(self, op, out, x):
        self.append("{}.fill({})", x, op.scalar)

    @generate_op.on_type(greater)
    def generate_op(self, op, out, x, y):
        self.append("np.greater({}, {}, out={})", x, y, out)

    @generate_op.on_type(greater_equal)
    def generate_op(self, op, out, x, y):
        self.append("np.greater_equal({}, {}, out={})", x, y, out)

    @generate_op.on_type(less)
    def generate_op(self, op, out, x, y):
        self.append("np.less({}, {}, out={})", x, y, out)

    @generate_op.on_type(less_equal)
    def generate_op(self, op, out, x, y):
        self.append("np.less_equal({}, {}, out={})", x, y, out)

    @generate_op.on_type(log)
    def generate_op(self, op, out, x):
        self.append("np.log({}, out={})", x, out)

    @generate_op.on_type(max)
    def generate_op(self, op, out, x):
        if op.mode is 'copy':
            self.append("{}.__setitem__((), {})", out, x)
        else:
            self.append("np.max({}, {}, out={})", x, op.mode, out)

    @generate_op.on_type(maximum)
    def generate_op(self, op, out, x, y):
        self.append("np.maximum({}, {}, out={})", x, y, out)

    @generate_op.on_type(min)
    def generate_op(self, op, out, x):
        if op.mode is 'copy':
            self.append("{}.__setitem__((), {})", out, x)
        else:
            self.append("np.min({}, {}, out={})", x, op.mode, out)

    @generate_op.on_type(minimum)
    def generate_op(self, op, out, x, y):
        self.append("np.minimum({}, {}, out={})", x, y, out)

    @generate_op.on_type(multiply)
    def generate_op(self, op, out, x, y):
        self.append("np.multiply({}, {}, out={})", x, y, out)

    @generate_op.on_type(negative)
    def generate_op(self, op, out, x):
        self.append("np.negative({}, out={})", x, out)

    @generate_op.on_type(not_equal)
    def generate_op(self, op, out, x, y):
        self.append("np.not_equal({}, {}, out={})", x, y, out)

    @generate_op.on_type(onehot)
    def generate_op(self, op, out, o, x):
        self.append("""
        o = {o}
        x = {x}
        o[:] = 0
        for i in range(len(x)):
            o[x[i], i] = 1
        """, x=x, o=o)

    @generate_op.on_type(power)
    def generate_op(self, op, out, x, y):
        self.append("np.power({}, {}, out={}", x, y, out)

    @generate_op.on_type(PrintOp)
    def generate_op(self, op, out, x):
        if op.prefix is not None:
            self.append("""
                print {prefix} + ':', {x}
                {out}[()] = {x}
            """, out=out, x=x, prefix=repr(op.prefix))
        else:
            self.append("""
                print {x}
                {out}[()] = {x}
            """, out=out, x=x)

    @generate_op.on_type(reciprocal)
    def generate_op(self, op, out, x):
        self.append("np.reciprocal({}, out={})", x, out)

    @generate_op.on_type(SetItem)
    def generate_op(self, op, out, tensor, value):
        self.append("{}.__setitem__({}, {})", tensor, op.item, value)

    @generate_op.on_type(sign)
    def generate_op(self, op, out, x):
        self.append("np.sign({}, out=out)", x, out)

    @generate_op.on_type(sin)
    def generate_op(self, op, out, x):
        self.append("np.sin({}, out={})", x, out)

    @generate_op.on_type(sqrt)
    def generate_op(self, op, out, x):
        self.append("np.sqrt({}, out={})", x, out)

    @generate_op.on_type(square)
    def generate_op(self, op, out, x):
        self.append("np.square({}, out={})", x, out)

    @generate_op.on_type(subtract)
    def generate_op(self, op, out, x, y):
        self.append("np.subtract({}, {}, out={})", x, y, out)

    @generate_op.on_type(sum)
    def generate_op(self, op, out, x):
        if op.mode is 'copy':
            self.append("{}.__setitem__((), {})", out, x)
        else:
            self.append("np.sum({}, axis={}, out={})", x, op.mode, out)

    @generate_op.on_type(tanh)
    def generate_op(self, op, out, x):
        self.append("np.tanh({}, out={})", x, out)

    @generate_op.on_type(tensor_size)
    def generate_op(self, op, out):
        self.append("{}.fill({})", out, op.reduction_axes.size)

    @generate_op.on_type(Unslice)
    def generate_op(self, op, out, out_sliced, x):
        self.append("{}.fill(0)", out)
        self.append("{}.__setitem__((), {})", out_sliced, x)

    @generate_op.on_type(Stack)
    def generate_op(self, op, out, *args):
        # TODO: we may want to have the inputs write into slices of a
        # preallocated buffer for this op.
        # We cannot use the numpy stack function as it is unavailable in
        # older versions.
        self.append("o={}", out)
        slices = [slice(None)] * len(op.axes)
        for i, arg in enumerate(args):
            slices[op.pos] = i
            self.append("o.__setitem__({s}, {x})", s=tuple(slices), x=arg)

    @generate_op.on_type(dimshuffle)
    def generate_op(self, op, out, x):
        # self.append("{out}[()] = {x}", out=out, x=x)
        self.append(
            "{out}[()] = np.transpose({x}, {axes})",
            out=out,
            x=x,
            axes=op.axes_order
        )


class NumPyTransformer(Transformer):
    """
    Transformer for executing graphs on a CPU, backed by numpy.

    Given a list of ops you want to compute the results of, this transformer
    will compile the graph required to compute those results and exposes an
    evaluate method to execute the compiled graph.
    """
    def __init__(self, **kwargs):
        super(NumPyTransformer, self).__init__(**kwargs)
        self.init_code = NumPyCodeGenerator()
        self.allocate_storage_code = NumPyCodeGenerator()
        self.allocate_code = NumPyCodeGenerator()
        self.compute_code = NumPyCodeGenerator()
        self.code = NumPyCodeGenerator()
        self.model = None
        self.n_computations = 0

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

    def transform_ordered_ops(self, ordered_ops):
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
                # TODO: handle case where TensorDescription has None value
                self.compute_code.generate_op(op, out, *call_info)
            if code is self.compute_code.code:
                self.compute_code.append("pass")
        self.compute_code.endl()
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
            self.code.append(self.allocate_storage_code.code)
            self.code.endl()
            if len(self.device_buffers) == 0:
                self.allocate_code.append("pass")
            self.code.append(self.allocate_code.code)
            self.code.endl(2)
            self.code.append(self.compute_code.code)

        r = self.code.compile("op", globals())
        self.model = r['Model']()
        for computation in self.computations:
            executor = getattr(self.model, computation.name)
            computation.executor = executor

    def allocate_storage(self):
        self.model.allocate()
