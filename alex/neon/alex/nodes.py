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

import neon.alex.ntypes as ntypes

class Node(object):
    def __init__(self, op_name=None, node_name=None, node_type=None, node_inputs=()):
        self.op_name = op_name
        self.node_name = node_name
        self.node_type = node_type
        self.node_inputs = node_inputs
        self.has_types = False

    @staticmethod
    def node(value):
        """Try to return a Node representing value."""
        if isinstance(value, Node):
            return value

        raise NotImplementedError()

    def infer_types(self):
        """Try to determine all relevant node types."""
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


class DataTensor(Node):
    def __init__(self, shape, dtype=None, name=None):
        super(DataTensor, self).__init__(op_name='DataTensor', node_type=ntypes.TensorType(shape=shape, dtype=dtype), node_name=name)

    def infer_types(self):
        if not self.has_types:
            self.has_type = True
        return True


data_tensor = DataTensor


class VariableTensor(Node):
    def __init__(self, shape, dtype=None, name=None):
        super(VariableTensor, self).__init__(op_name='VariableTensor', node_type=ntypes.TensorType(shape=shape, dtype=dtype), node_name=name)

    def infer_types(self):
        if not self.has_types:
            self.has_type = True
        return True


variable_tensor=VariableTensor


class UnaryNode(Node):
    def __init__(self, op_name, x, **kargs):
        super(UnaryNode, self).__init__(op_name=op_name, node_inputs=(x,), **kargs)

    def infer_types(self):
        pass


class SameTypeNode(object):
    def infer_types(self):
        if self.has_types:
            return True
        x = self.node_inputs
        if x.infer_types():
            if None == self.node_type:
                self.node_type = x.node_type.clone()
                self.has_types = True
            else:
                if x.node_type == self.node_type:
                    self.has_types = True
                else:
                    raise(ntypes.IncompatibleNodeTypes())

        elif None != self.node_type:
            x.node_type = self.node_type.clone()
            self.has_types = True

        return self.has_types


class Neg(UnaryNode, SameTypeNode):
    def __init__(self, x):
        super(Neg, self).__init__('Neg', x)


class Abs(UnaryNode, SameTypeNode):
    def __init__(self, x):
        super(Abs, self).__init__('Abs', x, node_type=x.node_type)


class BinaryNode(Node):
    def __init__(self, op_name, x, y, **kargs):
        super(BinaryNode, self).__init__(op_name=op_name, node_inputs=(x,y), **kargs)


class SumTypeNode(object):
    def infer_types(self):
        if self.has_types:
            return True
        x, y = self.node_inputs
        if x.infer_types() and y.infer_types():
            t1 = x.node_type
            t2 = y.node_type
            if not isinstance(t1, ntypes.TensorType) or not isinstance(t2, ntypes.TensorType):
                raise ntypes.IncompatibleNodeTypes()
            shape = ntypes.sum_broadcast_shape(t1.shape, t2.shape)
            self.node_type = ntypes.TensorType(shape=shape)
            self.has_types = True
        return self.has_types


class Add(BinaryNode, SumTypeNode):
    def __init__(self, x, y):
        super(Add, self).__init__('Add', x, y)


class Sub(BinaryNode, SumTypeNode):
    def __init__(self, x, y):
        super(Sub, self).__init__('Sub', x, y)


class Mul(BinaryNode):
    """Multiplies, but need to find out dim mapping.

    """
    def __init__(self, x, y):
        super(Mul, self).__init__('Mul', x, y)

    def infer_types(self):
        if self.has_types:
            return True
        x, y = self.node_inputs
        if x.infer_types() and y.infer_types():
            pass




class Div(BinaryNode):
    def __init__(self, x, y):
        super(Div, self).__init__('Div', x, y)


class Pow(BinaryNode):
    def __init__(self, x, y):
        super(Pow, self).__init__('Pow', x, y)

def windowed_size(window_size, input_size, pre_padding=0, post_padding=0, stride=1):
    """Return the number of outputs for given window, input, stride, and padding."""
    extended_size = pre_padding+input_size+post_padding
    return (extended_size-window_size)//stride+1

def padding_size(window_size):
    """Return padding needed for a given window size"""
    return (window_size-1)//2


class Conv(BinaryNode):
    """Primitive convolution node.

    Input is (C, T, H, W) for TxHxW -> Re^C
    Filter tensor is (C, T, R, S, K) which is K (C, T, R, S) filters
    stride is (s1, s2, s3), where s1=1
    padding is (p1, p2, p3) where pi is None for no padding, or 0 for 0 padding
    output is (K, O, P, Q) for OxPxQ -> Re^K
    """
    def __init__(self, x, filter, stride, padding, count):
        window = VariableTensor(filter+(count,))
        super(Conv, self).__init__('Conv', x, window)
        self.filter = filter
        self.stride = stride
        self.padding = padding
        self.count = count


conv=Conv


class MaxPool(UnaryNode):
    def __init__(self, x, filter, stride, padding):
        super(MaxPool, self).__init__('MaxPool', x)
        self.filter = filter
        self.stride = stride
        self.padding = padding


max_pool = MaxPool


class Relu(UnaryNode):
    def __init__(self, x):
        super(Relu, self).__init__('Relu', x, node_type=x.node_type)


relu = Relu