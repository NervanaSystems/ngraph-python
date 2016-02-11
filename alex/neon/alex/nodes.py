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

class Node(object):
    def __init__(self, op_name=None, node_name=None, node_type=None, node_inputs=()):
        self.op_name = op_name
        self.node_name = node_name
        self.node_type = node_type
        self.node_inputs = node_inputs


    @staticmethod
    def node(value):
        """Try to return a Node representing value."""
        if isinstance(value, Node):
            return value

        raise NotImplementedError()

    # Magic methods for builtin operations we want to use for creating nodes
    def __neg__(self):
        return Neg(self)

    def __pos__(self):
        return self

    def __abs__(self):
        return Abs(self)

    def __add__(self, val):
        return Add(self, val)

    def __radd__(self, val):
        return Add(val, self)

    def __sub__(self, val):
        return Sub(self, val)

    def __rsub__(self, val):
        return Sub(val, self)

    def __mul__(self, val):
        return Mul(self, val)

    def __rmul__(self, val):
        return Mul(val, self)

    def __div__(self, val):
        return Div(self, val)

    def __rdiv__(self, val):
        return Div(val, self)

    def __pow__(self, val):
        return Pow(self, val)

    def __rpow__(self, val):
        return Pow(self, val)


class NodeType(object):
    def __init__(self, name):
        self.name = name


class TensorType(NodeType):
    def __init__(self, shape, dtype=None):
        super(TensorType, self).__init__('TensorType')
        self.shape = shape
        self.dtype = dtype


class DataTensor(Node):
    def __init__(self, shape, dtype, name):
        super(DataTensor, self).__init__(op_name='DataTensor', node_type=TensorType(shape=shape, dtype=dtype), node_name=name)


def data_tensor(shape, dtype=None, name=None):
    return DataTensor(shape=shape, dtype=dtype, name=name)


class VariableTensor(Node):
    def __init__(self, shape, dtype, name):
        super(VariableTensor, self).__init__(op_name='VariableTensor', node_type=TensorType(shape=shape, dtype=dtype), node_name=name)


def variable_tensor(shape, dtype=None, name=None):
    return VariableTensor(shape=shape, dtype=dtype, name=name)


class UnaryNode(Node):
    def __init__(self, op_name, x, **kargs):
        super(UnaryNode, self).__init__(op_name=op_name, node_inputs=(x,), **kargs)


class Neg(UnaryNode):
    def __init__(self, x):
        super(Neg, self).__init__('Neg', x, node_type=x.node_type)


class Abs(UnaryNode):
    def __init__(self, x):
        super(Abs, self).__init__('Abs', x, node_type=x.node_type)


class BinaryNode(Node):
    def __init__(self, op_name, x, y, **kargs):
        super(BinaryNode, self).__init__(op_name=op_name, node_inputs=(x,y), **kargs)


class Add(BinaryNode):
    def __init__(self, x, y):
        super(Add, self).__init__('Add', x, y)


class Sub(BinaryNode):
    def __init__(self, x, y):
        super(Sub, self).__init__('Sub', x, y)


class Mul(BinaryNode):
    def __init__(self, x, y):
        super(Mul, self).__init__('Mul', x, y)


class Div(BinaryNode):
    def __init__(self, x, y):
        super(Div, self).__init__('Div', x, y)


class Pow(BinaryNode):
    def __init__(self, x, y):
        super(Pow, self).__init__('Pow', x, y)


class Conv(BinaryNode):
    def __init__(self, x, window, stride, pad):
        super(Conv, self).__init__('Conv', x, window)
        self.stride = stride
        self.pad = pad

def conv(x, weights, stride, pad):
    return Conv(x, weights, stride, pad)


class MaxPool(UnaryNode):
    def __init__(self, x, size, stride, pad):
        super(MaxPool, self).__init__('MaxPool', x)
        self.size = size
        self.stride = stride
        self.pad = pad


def max_pool(x, size, stride, pad):
    return MaxPool(x, size, stride, pad)


class Relu(UnaryNode):
    def __init__(self, x):
        super(Relu, self).__init__('Relu', x, node_type=x.node_type)


def relu(x):
    return Relu(x)
