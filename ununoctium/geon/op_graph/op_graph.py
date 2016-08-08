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

import numbers

import numpy as np
from builtins import object, str

from geon.backends.graph.environment import get_current_environment,\
    get_current_ops, captured_ops
from geon.op_graph.arrayaxes import get_batch_axes, TensorDescription, \
    AxisIDTuple, Axes, AxesAxis, Axis, sample_axes, batch_axes
from geon.op_graph.nodes import Node, generic_method


def tds(args, transformer):
    def td(arg):
        if isinstance(arg, Tensor):
            return arg.tensor_description(transformer)
        else:
            return None
    return (td(arg) for arg in args)


def from_transformer_cache(f):
    def wrapper(self, transformer, *args, **kargs):
        key = (f.__name__, self)
        if key not in transformer.cache:
            transformer.cache[key] = f(self, transformer, *args, **kargs)
        return transformer.cache[key]
    return wrapper


class Op(Node):
    """Any operation that can be in an AST"""

    def __init__(self, initializers=None, **kwds):
        super(Op, self).__init__(**kwds)
        self.schemas = []
        self._adjoints = None
        self.initializers = initializers or []
        ops = get_current_ops()
        if ops is not None:
            ops.append(self)

        self.transform_hook = None

    def add_schema(self, schema, set_generate_adjoints=True):
        """
        Adds a description of some op substructure.

        When a function generates a groups of nodes, it can add a schema
        describing the roles of these nodes.  The schema may include its
        own generate_adjoints.
        :param schema:
        :param set_generate_adjoints: Whether to override the node's generate_adjoints
         with the version from the schema.
        :return:
        """
        self.schemas.insert(0, schema)
        if set_generate_adjoints:
            # generate_adjoints is normally called with *args, but for a
            # schema we call it with the associated node.
            def generate_adjoints(adjoints, adjoint, *ignore):
                schema.generate_adjoints(adjoints, adjoint, self)
            # Replace generate_adjoints for self
            self.generate_adjoints = generate_adjoints

    def find_schema(self, t):
        """
        Find a schema of particular type.

        Searches added schema for one of type t.
        :param t: The type of schema desired.
        :return: A schema of type t, or None.
        """
        for schema in self.schemas:
            if isinstance(schema, t):
                return schema
        return None

    @property
    def defs(self):
        return {}

    def variables(self, trainable=True, filter=None):
        """Return all parameters used in computing this node"""
        params = set()

        if filter is None:
            filter = lambda node: ('trainable' in node.tags) is trainable

        def visitor(node):
            if isinstance(node, Variable) and filter(node):
                params.add(node)

        Node.visit_input_closure([self], visitor)

        return set(params)

    @staticmethod
    def get_ordered_ops(op, ordered_ops):
        """
        Get dependent ops ordered for autodiff.
        """
        Node.visit_input_closure([op], lambda o: ordered_ops.append(o))

    def adjoints(self):
        if self._adjoints is not None:
            return self._adjoints

        if len(self.axes) == 0:
            initial_adjoint = Constant(1)
        else:
            initial_adjoint = placeholder(axes=self.axes)
        self.initial_adjoint = initial_adjoint

        self._adjoints = dict()
        ordered_ops = []
        Op.get_ordered_ops(self, ordered_ops)
        self._adjoints[self] = self.initial_adjoint
        for o in list(reversed(ordered_ops)):
            if o in self._adjoints:
                scale = o.scale
                adjoint = self._adjoints[o]
                if scale is not None:
                    adjoint = adjoint * scale
                o.generate_adjoints(self._adjoints, adjoint, *o.args)
        return self._adjoints

    @staticmethod
    def ordered_ops(results):
        ordered_ops = []
        Node.visit_input_closure(
            results, lambda o: ordered_ops.append(o))
        return ordered_ops

    def as_node(self, x):
        return Op.as_op(x)

    @staticmethod
    def as_op(x):
        if isinstance(x, Tensor):
            return x

        return Constant(x)

    @staticmethod
    def as_ops(xs):
        return tuple(Op.as_op(x) for x in xs)

    @property
    def ops(self):
        return []

    @staticmethod
    def simple_prune(results):
        SimplePrune(results)

    def transform(self, transformer, *args):
        """Process op"""
        pass

    def sync(self, transformer):
        """Make sure transformer has local changes"""
        pass

    def allocate(self, transformer):
        pass

    @from_transformer_cache
    def call_info(self, transformer):
        return list(tds(self.args, transformer))

    def create_views(self, transformer):
        self.call_info(transformer)

    def transform_call_info(self, transformer):
        def value(arg):
            if arg is None:
                return None
            else:
                return arg.value
        call_args = [value(arg) for arg in self.call_info(transformer)]
        self.transform(transformer, *call_args)

    def __str__(self):
        return '<{cl}:{id}>'.format(cl=self.__class__.__name__, id=id(self))


class Tensor(Op):

    def __init__(
            self, dtype=None, axes=None, scale=None,
            out=None, has_alloc=False, **kwds):
        super(Tensor, self).__init__(**kwds)
        if dtype is None:
            dtype = np.dtype(np.float32)
        self.dtype = dtype
        if axes is not None:
            axes = Axes(*axes)
        self.__axes = axes
        self.__out = out
        self.has_alloc = has_alloc

        # Derivative will be scaled by this
        self.scale = scale

    @property
    def output(self):
        if self.__out is not None:
            return self.__out
        else:
            return self

    def generate_add_delta(self, adjoints, delta):
        delta = sum(delta, reduction_axes=delta.axes - self.axes)
        if self not in adjoints:
            adjoints[self] = delta
        else:
            adjoints[self] = delta + adjoints[self]

    # Magic methods for builtin operations we want to use for creating nodes
    def __neg__(self):
        return negative(self)

    def __pos__(self):
        return self

    def __abs__(self):
        return absolute(self)

    def __add__(self, val):
        return add(self, val)

    def __radd__(self, val):
        return add(val, self)

    def __sub__(self, val):
        return subtract(self, val)

    def __rsub__(self, val):
        return subtract(val, self)

    def __mul__(self, val):
        return multiply(self, val)

    def __rmul__(self, val):
        return multiply(val, self)

    def __truediv__(self, val):
        return divide(self, val)

    def __rtruediv__(self, val):
        return divide(val, self)

    def __rdiv__(self, val):
        return divide(val, self)

    def __pow__(self, val):
        return power(self, val)

    def __rpow__(self, val):
        return power(val, self)

    # Python always uses eq for comparing keys, so if we override __eq__ we
    # cannot have sets of tensors, or using them as dictionary keys.  So,
    # we must use Equal explicitly in transfrom.  defmod and define __eq__
    # if it can ensure that its nodes do not need to be used as keys.
    # def __eq__(self, val):
    #    return equal(self, val)

    # def __ne__(self, val):
    #    return not_equal(self, val)

    def __lt__(self, val):
        return less(self, val)

    def __gt__(self, val):
        return greater(self, val)

    def __le__(self, val):
        return less_equal(self, val)

    def __ge__(self, val):
        return greater_equal(self, val)

    # Only works when capturing ops
    def __setitem__(self, key, val):
        return SetItem(self, key, val)

    # Only works when capturing ops
    def __iadd__(self, val):
        return SetItem(self, slice(None, None, None), self + val)

    # Only works when capturing ops
    def __isub__(self, val):
        return SetItem(self, slice(None, None, None), self - val)

    # Only works when capturing ops
    def __imul__(self, val):
        return SetItem(self, slice(None, None, None), self * val)

    # Only works when capturing ops
    def __idiv__(self, val):
        return SetItem(self, slice(None, None, None), self / val)

    def __axes__(self):
        return self.axes

    def output_value(self, transformer):
        return self.tensor_description(transformer).value

    @from_transformer_cache
    def tensor_description(self, transformer):
        if self.__out is not None:
            td = self.__out.tensor_description(transformer)
            if self.__axes is not None:
                td = td.reaxe(self.__axes)
        else:
            td = TensorDescription(self.axes, transformer, dtype=self.dtype)
        return td

    def create_views(self, transformer):
        self.tensor_description(transformer)
        self.call_info(transformer)

    @property
    def axes(self):
        if self.__axes is not None:
            return self.__axes
        elif self.__out is not None:
            return self.__out.axes
        else:
            raise NotImplementedError

    def allocate(self, transformer):
        if self.__out is None:
            td = self.tensor_description(transformer)
            buffer = td.buffer
            if buffer.data is None:
                buffer.data = transformer.make_raw_buffer(buffer.size)
            if self.has_alloc:
                tensor = self.allocator(transformer)
            else:
                tensor = transformer.tensor_view(td)
            td.value = tensor

    def allocator(self, transformer):
        return transformer.empty(self.tensor_description(transformer))

    def generate_adjoints(self, *args, **kargs):
        pass

    @from_transformer_cache
    def call_info(self, transformer):
        return [self.tensor_description(transformer)]\
            + super(Tensor, self).call_info(transformer)

    # Required for parameter initializers
    @property
    def shape(self):
        return self.axes

    def mean(self, out_axes=(), **kargs):
        return mean(self, out_axes=out_axes, **kargs)


class Broadcast(Tensor):
    """
    Used to add additional axes for a returned derivative.

    """

    def __init__(self, x, **kargs):
        super(Broadcast, self).__init__(args=(x,), out=x, **kargs)


class ExpandDims(Tensor):

    def __init__(self, x, axis, dim, **kargs):
        axes = x.axes[:dim].concat(Axes(axis,)).concat(x.axes[dim:])
        super(ExpandDims, self).__init__(args=(x,), axes=axes, **kargs)
        self.axis = axis
        self.dim = dim

    @from_transformer_cache
    def tensor_description(self, transformer):
        x, = tds(self.args, transformer)
        return x.reaxe_with_dummy_axis(self.axis, self.dim)

    def generate_adjoints(self, adjoints, delta, x):
        x.generate_add_delta(
            adjoints,
            sum(delta, reduction_axes=Axes(self.axis,))
        )


class AllocationOp(Tensor):

    def __init__(
            self,
            init=None,
            initial_value=None,
            **kargs):
        super(AllocationOp, self).__init__(has_alloc=True, **kargs)
        if init is not None:
            with captured_ops(self.initializers):
                init.fill(self)
        elif callable(initial_value):
            self.initializers.append(assign(self, initial_value()))
        elif initial_value is not None:
            self.initializers.append(assign(self, initial_value))


class ComputationOp(Tensor):
    """
    An TensorOp is the result of some sort of operation.
    """

    def __init__(self, dtype=np.dtype(np.float32), batch_axes=None, **kargs):
        super(ComputationOp, self).__init__(**kargs)
        self.dtype = dtype

        for arg in self.args:
            arg.users.add(self)

        if batch_axes is None:
            batch_axes = get_batch_axes()
        self.batch_axes = batch_axes

    @property
    def defs(self):
        return {self}

    @property
    def graph_label(self):
        return self.__class__.__name__ + '[' + self.name + ']'


class RNG(object):

    def __init__(self, seed=None):
        self.seed = seed
        self.rng = np.random.RandomState(seed=seed)

    def uniform(self, low=0.0, high=1.0, size=None, **kargs):
        return Uniform(rng=self.rng, low=low, high=high, size=size, **kargs)

    def normal(self, loc, scale, size, **kargs):
        return Normal(rng=self.rng, loc=loc, scale=scale, size=size, **kargs)


class RNGOp(AllocationOp):

    def __init__(self, rng, **kargs):
        super(RNGOp, self).__init__(**kargs)
        self.rng = rng


class Normal(RNGOp):

    def __init__(self, loc=0.0, scale=1.0, size=None, **kargs):
        super(Normal, self).__init__(axes=size, **kargs)
        self.loc = loc
        self.scale = scale

    def allocator(self, transformer):
        td, rng = self.call_info(transformer)
        return transformer.rng_normal_tensor(
            rng.value, td,
            self.loc, self.scale
        )


class Uniform(RNGOp):

    def __init__(self, low=0.0, high=1.0, size=None, **kargs):
        super(Uniform, self).__init__(axes=size, **kargs)
        self.low = low
        self.high = high

    def allocator(self, transformer):
        td, rng, = self.call_info(transformer)
        return transformer.rng_uniform_tensor(
            rng.value, td,
            self.low, self.high
        )


class VoidOp(ComputationOp):

    def __init__(self, **kargs):
        super(VoidOp, self).__init__(axes=Axes(), **kargs)


class SetItem(VoidOp):

    def __init__(self, tensor, item, val, **kargs):
        super(SetItem, self).__init__(args=(tensor, val), out=tensor, **kargs)
        self.item = item

    @from_transformer_cache
    def call_info(self, transformer):
        tensor, val = tds(self.args, transformer)
        return [tensor, val.reaxe(self.axes)]

    def transform(self, transformer, tensor, val):
        transformer.set_item(tensor, self.item, val)


class doall(VoidOp):

    def __init__(self, all, **kargs):
        super(doall, self).__init__(args=all, out=all[-1], **kargs)


class ElementWise(ComputationOp):

    def __init__(self, args, **kargs):
        args = Op.as_ops(args)
        axis_ids = AxisIDTuple()
        for arg in args:
            axis_ids += arg.axes.as_axis_ids()
        axes = axis_ids.as_axes()
        super(ElementWise, self).__init__(
            args=args,
            axes=axes,
            **kargs
        )

    @from_transformer_cache
    def call_info(self, transformer):
        ci = [self.tensor_description(transformer)]
        for arg in tds(self.args, transformer):
            ci.append(arg.reaxe(self.axes))
        return ci


class AllReduce(ElementWise):

    def __init__(self, x, **kargs):
        super(AllReduce, self).__init__(args=(x,), **kargs)

    def transform(self, transformer, out, x):
        return transformer.allreduce(x, out)


class placeholder(AllocationOp):
    """
    Can be set externally.
    """

    def __init__(self, tags=None, **kargs):
        if tags is None:
            tags = set()
        tags.add('persistent')
        super(placeholder, self).__init__(**kargs)

    @property
    def value(self):
        return get_current_environment()[self]

    @value.setter
    def value(self, value):
        get_current_environment()[self] = value

    def sync(self, transformer):
        value = self.value
        td = self.tensor_description(transformer)
        if isinstance(value, numbers.Real):
            transformer.fill(td.value, value)
        else:
            td.value = value


class Fill(VoidOp):

    def __init__(self, tensor, const, **kargs):
        super(Fill, self).__init__(args=(tensor,), **kargs)
        self.const = const

    def transform(self, transformer, tensor):
        transformer.fill(tensor, self.const)


class Constant(AllocationOp):
    """
    A scalar constant that appears in a graph.
    """

    def __init__(self, const, **kargs):
        self.const = const
        super(Constant, self).__init__(
            axes=(), dtype=np.dtype(np.float32), **kargs)
        self.initializers.append(Fill(self, const))
        self.tags.add('persistent')

    @property
    def graph_label(self):
        shapes = self.tensor_axes_info.tensor_description.shape
        if not shapes or max(shapes) <= 2:
            return str(self.const)
        if self.name == self.id:
            return 'Constant'
        return self.name

    def __str__(self):
        return '<{cl} ({const})>'.format(
            cl=self.__class__.__name__, const=self.const)


class NumPyTensor(AllocationOp):
    """
    A NumPy tensor with attached axes information
    """

    def __init__(self, nptensor, **kargs):
        self.nptensor = nptensor
        super(NumPyTensor, self).__init__(
            dtype=nptensor.dtype, **kargs
        )

    def allocator(self, transformer):
        return transformer.nparray(
            self.tensor_description(transformer), self.nptensor
        )

    @property
    def graph_label(self):
        return str(self.nptensor.shape)

    def __str__(self):
        return '<{cl} ({const})>'.format(
            cl=self.__class__.__name__, const=self.nptensor)


class absolute(ElementWise):

    def __init__(self, x, **kargs):
        super(absolute, self).__init__(args=(x,), **kargs)

    def transform(self, transformer, out, x):
        transformer.absolute(x, out)

    def generate_adjoints(self, adjoints, delta, x):
        x.generate_add_delta(adjoints, sgn(x) * delta)


class add(ElementWise):

    def __init__(self, x, y, **kargs):
        super(add, self).__init__(args=(x, y), **kargs)

    def transform(self, transformer, out, x, y):
        transformer.add(x, y, out)

    def generate_adjoints(self, adjoints, delta, x, y):
        x.generate_add_delta(adjoints, delta)
        y.generate_add_delta(adjoints, delta)


class argmax(ComputationOp):
    def __init__(self, x, max_axes=None, **kargs):
        if max_axes is None:
            max_axes = sample_axes(x.axes)
            axes = batch_axes(x.axes)
        else:
            axes = x.axes - max_axes
        self.max_axes = max_axes
        super(argmax, self).__init__(
            args=(x,), axes=axes, dtype=np.dtype(np.int64), **kargs
        )

    @from_transformer_cache
    def call_info(self, transformer):
        x, = tds(self.args, transformer)
        return [
            self.tensor_description(transformer),
            x.reaxe(self.max_axes + self.axes)
        ]

    def transform(self, transformer, out, x):
        transformer.argmax(x, out)


class argmin(ComputationOp):

    def __init__(self, x, min_axes=None, **kargs):
        if min_axes is None:
            min_axes = sample_axes(x.axes)
            axes = batch_axes(x.axes)
        else:
            axes = x.axes - min_axes
        self.min_axes = min_axes
        super(argmax, self).__init__(
            args=(x,), axes=axes, dtype=np.dtype(np.int64), **kargs
        )

    @from_transformer_cache
    def call_info(self, transformer):
        x, = tds(self.args)
        return [
            self.tensor_description(transformer),
            x.reaxe(self.min_axes + self.axes)
        ]

    def transform(self, transformer, out, x):
        transformer.argmin(x, out)


class cos(ElementWise):

    def __init__(self, x, **kargs):
        super(cos, self).__init__(args=(x,), **kargs)

    def generate_adjoints(self, adjoints, delta, x):
        x.generate_add_delta(adjoints, delta * sin(x))

    def transform(self, transformer, out, x):
        transformer.cos(x, out)


class divide(ElementWise):

    def __init__(self, x, y, **kargs):
        super(divide, self).__init__(args=(x, y), **kargs)

    def transform(self, transformer, out, x, y):
        transformer.divide(x, y, out)

    def generate_adjoints(self, adjoints, delta, x, y):
        x.generate_add_delta(adjoints, delta * self / x)
        y.generate_add_delta(adjoints, -delta * self / y)


class dot(ComputationOp):

    def __init__(self, x, y,
                 reduction_axes=None, out_axes=None,
                 numpy_matching=False,
                 forward_dot=None,
                 **kargs):
        self.axis_id_info = self.compute_axis_id_info(
            x, y, reduction_axes, out_axes,
            forward_dot, numpy_matching
        )
        self.out_axes = out_axes
        self.reduction_axes = reduction_axes
        axes = self.axis_id_info[0].as_axes()
        super(dot, self).__init__(
            args=(x, y), axes=axes, **kargs
        )

    def compute_axis_id_info(self, x, y,
                             reduction_axes, out_axes,
                             forward_dot, use_numpy_matching):
        x_axis_ids = x.axes.as_axis_ids()
        y_axis_ids = y.axes.as_axis_ids()

        if forward_dot is not None:
            y_axis_ids = forward_dot.axis_id_info[0]
            forward_axis_ids = forward_dot.axis_id_info[0]
        else:
            forward_axis_ids = None

        if use_numpy_matching:
            out_axis_ids = x_axis_ids[:-1]\
                + y_axis_ids[:-2]\
                + AxisIDTuple(y_axis_ids[-1],)
            x_red_axis_ids = AxisIDTuple(x_axis_ids[-1])
            y_red_axis_ids = AxisIDTuple(y_axis_ids[-2])
            return (out_axis_ids, x_red_axis_ids, y_red_axis_ids,
                    None, forward_axis_ids)
        else:
            dummy = None
            if reduction_axes is None:
                red_axis_ids = AxisIDTuple.intersect(
                    x_axis_ids,
                    y_axis_ids
                )
            else:
                red_axis_ids = reduction_axes.as_axis_ids()

            if out_axes is not None:
                out_axis_ids = out_axes.as_axis_ids()
            else:
                out_axis_ids = (
                    (x_axis_ids - red_axis_ids) +
                    (y_axis_ids - red_axis_ids)
                )
            red_axis_ids -= out_axis_ids

            if len(red_axis_ids) == 0:
                dummy = Axis(1)
                red_axis_ids = AxisIDTuple(dummy[0],)
            return (out_axis_ids, red_axis_ids, red_axis_ids,
                    dummy, forward_axis_ids)

    @from_transformer_cache
    def call_info(self, transformer):
        x, y = tds(self.args, transformer)
        out_axis_ids, x_red_axis_ids, y_red_axis_ids, dummy, forward_axis_ids\
            = self.axis_id_info

        if dummy is not None:
            x = x.reaxe_with_dummy_axis(dummy)
            y = y.reaxe_with_dummy_axis(dummy)

        a = x.dot_reaxe_left(x_red_axis_ids)
        b = y.dot_reaxe_right(
            y_red_axis_ids,
            forward_axis_ids=forward_axis_ids
        )
        a_axes, b_axes = a.axes, b.axes
        o = self.tensor_description(transformer)\
            .reaxe(a_axes[:-1].concat(b_axes[1:]))

        return [o, a, b]

    def transform(self, transformer, out, x, y):
        transformer.dot(x, y, out)

    def generate_adjoints(self, adjoints, delta, x, y):
        # The delta must be passed in as the second argument
        # to preserve the forward axes mapping.
        x.generate_add_delta(
            adjoints,
            dot(y, delta, out_axes=x.axes, forward_dot=self)
        )
        y.generate_add_delta(
            adjoints,
            dot(x, delta, out_axes=y.axes, forward_dot=self)
        )


class ElementWiseBoolean(ElementWise):

    def __init__(self, x, y, dtype=np.dtype(bool), **kargs):
        super(ElementWiseBoolean, self).__init__(
            args=(x, y), dtype=dtype, **kargs)


class equal(ElementWiseBoolean):

    def transform(self, transformer, out, x, y):
        transformer.equal(x, y, out)


class not_equal(ElementWiseBoolean):

    def transform(self, transformer, out, x, y):
        transformer.not_equal(x, y, out)


class greater(ElementWiseBoolean):

    def transform(self, transformer, out, x, y):
        transformer.greater(x, y, out)


class less(ElementWiseBoolean):

    def transform(self, transformer, out, x, y):
        transformer.less(x, y, out)


class greater_equal(ElementWiseBoolean):

    def transform(self, transformer, out, x, y):
        transformer.greater_equal(x, y, out)


class less_equal(ElementWiseBoolean):

    def transform(self, transformer, out, x, y):
        transformer.less_equal(x, y, out)


class Softmax(object):

    def __init__(self, x, exps, Z):
        self.x = x
        self.exps = exps
        self.Z = Z

    def generate_adjoints(self, adjoints, delta, op):
        z = delta * op
        zs = sum(z, reduction_axes=sample_axes(self.x.axes))
        self.x.generate_add_delta(adjoints, (z - zs * op))


def softmax(x, softmax_axes=None, **kargs):
    if softmax_axes is None:
        softmax_axes = sample_axes(x.axes)
    x = x - max(x, reduction_axes=softmax_axes)
    exps = exp(x)
    Z = sum(exps, reduction_axes=softmax_axes)
    result = exps / Z
    result.add_schema(Softmax(x=x, exps=exps, Z=Z))
    return result


class ReductionOp(ComputationOp):

    def __init__(self, x, reduction_axes=None, out_axes=None, **kargs):
        self.out_axes, self.reduction_axes\
            = self.compute_axes(x, reduction_axes, out_axes)
        self.mode = None
        super(ReductionOp, self).__init__(
            args=(x,), axes=self.out_axes, **kargs
        )

    def compute_axes(self, x, reduction_axes, out_axes):
        if reduction_axes is None:
            if out_axes is None:
                reduction_axes = sample_axes(x.axes)
            else:
                reduction_axes = x.axes - Axes(*out_axes)
        else:
            reduction_axes = Axes(*reduction_axes)
        if out_axes is None:
            out_axes = x.axes - reduction_axes
        else:
            out_axes = Axes(*out_axes)
        return out_axes, reduction_axes

    @from_transformer_cache
    def call_info(self, transformer):
        x, = tds(self.args, transformer)
        out = self.tensor_description(transformer)

        if len(self.reduction_axes) == 0:
            # TODO do this as a reaxe to 1d or something
            xr = x.reaxe(self.axes)
            self.mode = 'copy'
            return [out, xr]
        else:
            red_axes = [AxesAxis(self.reduction_axes)]
            red_axes.extend(self.axes)
            red_axes = Axes(*red_axes)
            self.mode = 0
            return [out, x.reaxe(red_axes)]


class max(ReductionOp):

    def __init__(self, x, **kargs):
        super(max, self).__init__(x, **kargs)

    def transform(self, transformer, out, x):
        if self.mode is 'copy':
            # TODO Change this to a node replace
            transformer.set_item(out, (), x)
        else:
            transformer.max(x, self.mode, out)

    def generate_adjoints(self, adjoints, delta, x):
        x.generate_add_delta(adjoints, equal(x, self) * delta)


class min(ReductionOp):

    def __init__(self, x, **kargs):
        super(min, self).__init__(x, **kargs)

    def transform(self, transformer, out, x):
        if self.mode is 'copy':
            # TODO Change this to a node replace
            transformer.set_item(out, (), x)
        else:
            transformer.min(x, self.mode, out)

    def generate_adjoints(self, adjoints, delta, x):
        x.generate_add_delta(adjoints, equal(x, self) * delta)


class sum(ReductionOp):

    def __init__(self, x, **kargs):
        super(sum, self).__init__(x, **kargs)

    def transform(self, transformer, out, x):
        if self.mode is 'copy':
            # TODO Change this to a node replace
            transformer.set_item(out, (), x)
        else:
            transformer.sum(x, self.mode, out)

    def generate_adjoints(self, adjoints, delta, x):
        x.generate_add_delta(adjoints, delta)


def assign(lvalue, rvalue):
    return SetItem(lvalue, slice(None, None, None), rvalue)


class tensor_size(ComputationOp):

    def __init__(self, x, reduction_axes=None, out_axes=None, **kargs):
        if reduction_axes is None:
            if out_axes is None:
                reduction_axes = sample_axes(x.axes)
            else:
                reduction_axes = x.axes - Axes(*out_axes)
        else:
            reduction_axes = Axes(*reduction_axes)
        self.reduction_axes = reduction_axes
        super(tensor_size, self).__init__(axes=Axes())

    def transform(self, transformer, out):
        transformer.fill(out, self.reduction_axes.size)


class Slice(ComputationOp):

    def __init__(self, slices, x, **kargs):
        super(Slice, self).__init__(args=(x,), **kargs)
        self.slices = slices


class Pad(ComputationOp):

    def __init__(self, axes, slice, x, **kargs):
        super(Pad, self).__init__(args=(x,), axes=axes, **kargs)
        self.slice = slice

    def transform(self, transformer, out, x):
        transformer.pad(x, self.slice, out)


class Variable(AllocationOp):

    def __init__(self, tags=None, trainable=True, persistent=True, **kargs):
        if tags is None:
            tags = set()
        else:
            tags = set(tags)
        if trainable:
            tags.add('trainable')
        if persistent:
            tags.add('persistent')
        super(Variable, self).__init__(tags=tags, **kargs)


def temporary(**kargs):
    return Variable(trainable=False, persistent=True, **kargs)


class exp(ElementWise):

    def __init__(self, x, **kargs):
        super(exp, self).__init__(args=(x,), **kargs)

    def generate_adjoints(self, adjoints, delta, x):
        x.generate_add_delta(adjoints, delta * self)

    def transform(self, transformer, out, x):
        transformer.exp(x, out)


class log(ElementWise):

    def __init__(self, x, **kargs):
        super(log, self).__init__(args=(x,), **kargs)

    def generate_adjoints(self, adjoints, delta, x):
        def do_adjoints(delta, x):

            if False and isinstance(x, softmax):
                x, = x.args

            if isinstance(x, divide):
                a, b = x.args
                do_adjoints(delta, a)
                do_adjoints(-delta, b)

            elif isinstance(x, exp):
                x.args[0].generate_add_delta(adjoints, delta)

            else:
                x.generate_add_delta(adjoints, delta / x)

        do_adjoints(delta, x)

    def transform(self, transformer, out, x):
        transformer.log(x, out)


class safelog(log):
    expm50 = np.exp(-50.)

    def transform(self, transformer, out, x):
        transformer.maximum(x, safelog.expm50, out)
        transformer.log(out, out)


class maximum(ElementWise):

    def __init__(self, x, y, **kargs):
        super(maximum, self).__init__(args=(x, y), **kargs)

    def transform(self, transformer, out, x, y):
        transformer.maximum(x, y, out=out)

    def generate_adjoints(self, adjoints, delta, x, y):
        x.generate_add_delta(adjoints, equal(self, x) * delta)
        y.generate_add_delta(adjoints, equal(self, y) * delta)


class minimum(ElementWise):

    def __init__(self, x, y, **kargs):
        super(minimum, self).__init__(args=(x, y), **kargs)

    def transform(self, transformer, out, x, y):
        transformer.minimum(x, y, out=out)

    def generate_adjoints(self, adjoints, delta, x, y):
        x.generate_add_delta(adjoints, equal(self, x) * delta)
        y.generate_add_delta(adjoints, equal(self, y) * delta)


class multiply(ElementWise):

    def __init__(self, x, y, **kargs):
        super(multiply, self).__init__(args=(x, y), **kargs)

    def generate_adjoints(self, adjoints, delta, x, y):
        x.generate_add_delta(adjoints, delta * y)
        y.generate_add_delta(adjoints, x * delta)

    def transform(self, transformer, out, x, y):
        transformer.multiply(x, y, out)


class negative(ElementWise):

    def __init__(self, x, **kargs):
        super(negative, self).__init__(args=(x,), **kargs)

    def generate_adjoints(self, adjoints, delta, x):
        x.generate_add_delta(adjoints, -delta)

    def transform(self, transformer, out, x):
        transformer.negative(x, out)


class onehot(ComputationOp):

    def __init__(self, x, axis=None, axes=None, **kargs):
        if axis is None:
            if axes is None:
                raise ValueError('Cannot determine one-hot axis.')
            axis = (axes - x.axes)[0]
        else:
            if axes is None:
                x_sample = sample_axes(x.axes)
                x_batch = batch_axes(x.axes)
                axes = Axes(axis) + x_sample + x_batch
        self.axis = axis
        super(onehot, self).__init__(args=(x,), axes=axes, **kargs)

    @from_transformer_cache
    def call_info(self, transformer):
        x, = tds(self.args, transformer)
        axis, axes = self.axis, self.axes
        reaxes = Axes(axis, AxisIDTuple.sub(axes, Axes(axis,)).as_axes())
        return [
            self.tensor_description(transformer).reaxe(reaxes),
            x.reaxe(Axes(AxesAxis(x.axes)))
        ]

    def transform(self, transformer, out, x):
        transformer.onehot(x, out)


class power(ElementWise):

    def __init__(self, x, y, **kargs):
        super(power, self).__init__(args=(x,), **kargs)

    def transform(self, transformer, out, x, y):
        transformer.pow(x, y, out)

    def generate_adjoints(self, adjoints, delta, x, y):
        x.generate_add_delta(adjoints, delta * y * self / x)
        y.generate_add_delta(adjoints, delta * self * log(x))


class reciprocal(ElementWise):

    def __init__(self, x, **kargs):
        super(reciprocal, self).__init__(args=(x,), **kargs)

    def generate_adjoints(self, adjoints, delta, x):
        x.generate_add_delta(adjoints, -self * self * delta)

    def transform(self, transformer, out, x):
        transformer.reciprocal(x, out)


class sgn(ElementWise):

    def __init__(self, x, **kargs):
        super(sgn, self).__init__(args=(x,), **kargs)

    def generate_adjoints(self, adjoints, delta, x):
        # Zero
        pass

    def transform(self, transformer, out, x):
        transformer.sign(x, out)


class Sig(object):
    """Sigmoid"""

    def __init__(self, x):
        self.x = x

    def generate_adjoints(self, adjoints, delta, op):
        self.x.generate_add_delta(adjoints, delta * op * (1.0 - op))


def sig(x, **kargs):
    result = reciprocal(exp(-x) + 1)
    result.add_schema(Sig(x=x))
    return result


class sin(ElementWise):

    def __init__(self, x, **kargs):
        super(sin, self).__init__(args=(x,), **kargs)

    def generate_adjoints(self, adjoints, delta, x):
        x.generate_add_delta(adjoints, delta * cos(x))

    def transform(self, transformer, out, x):
        transformer.sin(x, out)


class sqrt(ElementWise):

    def __init__(self, x, **kargs):
        super(sqrt, self).__init__(args=(x,), **kargs)

    def generate_adjoints(self, adjoints, delta, x):
        x.generate_add_delta(adjoints, .5 * delta * self)

    def transform(self, transformer, out, x):
        transformer.sqrt(x, out)


class square(ElementWise):

    def __init__(self, x, **kargs):
        super(square, self).__init__(args=(x,), **kargs)

    def generate_adjoints(self, adjoints, delta, x):
        x.generate_add_delta(adjoints, 2.0 * delta * x)

    def transform(self, transformer, out, x):
        transformer.square(x, out)


class subtract(ElementWise):

    def __init__(self, x, y, **kargs):
        super(subtract, self).__init__(args=(x, y), **kargs)

    def generate_adjoints(self, adjoints, delta, x, y):
        x.generate_add_delta(adjoints, delta)
        y.generate_add_delta(adjoints, -delta)

    def transform(self, transformer, out, x, y):
        transformer.subtract(x, y, out)


class tanh(ElementWise):

    def __init__(self, x, **kargs):
        super(tanh, self).__init__(args=(x,), **kargs)

    def generate_adjoints(self, adjoints, delta, x):
        x.generate_add_delta(adjoints, delta * (1.0 - self * self))

    def transform(self, transformer, out, x):
        transformer.tanh(x, out)


class Function(Node):

    def __init__(self, ops):
        super(Function, self).__init__()
        from geon.util.analysis import Digraph
        self.ops = Digraph(ops)
        self.instructions = self.ops.topsort()
        args, defs = set(), set()
        for op in self.instructions:
            # Kernel defines the def of each operation
            defs |= set([op])
            # Kernel uses the args of each operation
            # except whatever is being defined
            args |= set(op.args) - defs
        self.args = args
        self.defs = defs

    @property
    def inputs(self):
        return self.use


class Buffer(object):

    def __init__(self, color, size):
        self.color = color
        self.size = size
        self.data = None
        self.views = set()


def mean(x, **kargs):
    return sum(x, **kargs) / tensor_size(x, **kargs)


def deriv(dep, indep):
    Op.simple_prune([dep, indep])
    adjoint = dep.adjoints()[indep]
    if adjoint.axes == indep.axes:
        return adjoint
    else:
        return Broadcast(adjoint, axes=indep.axes)


class CrossEntropyMultiInner(object):

    def __init__(self, x, y, s):
        self.x = x
        self.y = y
        self.s = s

    def generate_adjoints(self, adjoints, delta, op):
        self.s.generate_add_delta(adjoints, delta)
        self.x.generate_add_delta(adjoints, self.y * delta)


def cross_entropy_multi(y, t, usebits=False,
                        out_axes=None, enable_softmax_opt=True,
                        enable_diff_opt=True, **kargs):
    smy = y.find_schema(Softmax)
    if enable_softmax_opt and smy is not None:
        # This depends on sum(t) being 1
        x = smy.x
        Z = smy.Z
        s = -sum(x * t, out_axes=out_axes)
        result = s + safelog(Z)
        if enable_diff_opt:
            result.add_schema(CrossEntropyMultiInner(x=x, y=y, s=s))
    else:
        result = -sum(safelog(y) * t, out_axes=out_axes)
    if usebits:
        result = result * np.float(1. / np.log(2.0))
    return result


class CrossEntropyBinaryInner(object):

    def __init__(self, x, y, t):
        self.x = x
        self.y = y
        self.t = t

    def generate_adjoints(self, adjoints, delta, op):
        self.x.generate_add_delta(adjoints, (self.y - self.t) * delta)
        self.t.generate_add_delta(adjoints, self.x * delta)


def cross_entropy_binary_inner(y, t, enable_sig_opt=True,
                               enable_diff_opt=True, **kargs):
    sigy = y.find_schema(Sig)
    if enable_sig_opt and sigy is not None:
        # Simpler equivalent
        x = sigy.x
        result = (1 - t) * x - safelog(y)
        if enable_diff_opt:
            result.add_schema(CrossEntropyBinaryInner(x=x, y=y, t=t))
    else:
        result = -(safelog(y) * t + safelog(1 - y) * (1 - t))

    return result


def cross_entropy_binary(y, t, out_axes=None):
    return sum(cross_entropy_binary_inner(y, t), out_axes=out_axes)


def set_break(op, name=None):
    def hook(transformer, op, transform_op):
        transform_op(op)
        # print(name)
        pass
    op.transform_hook = hook
    return op


class SimplePrune(object):

    def __init__(self, results):
        self.results = results
        self.reps = []

        has_work = True
        while has_work:
            self.init()
            for op in Op.ordered_ops(self.results):
                self.visit(op)
            has_work = self.do_replacements()

    def init(self):
        self.reps = []

    def add_rep(self, op, replacement):
        # Can't replace op if its being returned
        if op not in self.results:
            self.reps.append((op, replacement))

    def do_replacements(self):
        for old, rep in self.reps:
            old_users = set(old.users)
            for user in old_users:
                user.replace_arg(old, rep)
        return len(self.reps) > 0

    @generic_method
    def visit(self, op):
        pass

    @visit.on_type(negative)
    def visit(self, op):
        x, = op.args
        if isinstance(x, Constant):
            self.add_rep(op, Constant(-x.const))

    @visit.on_type(multiply)
    def visit(self, op):
        x, y = op.args
        rep = None
        if isinstance(x, Constant):
            if x.const == 0:
                rep = x
            elif x.const == 1:
                rep = y
            elif x.const == -1:
                rep = negative(y)
        elif isinstance(y, Constant):
            if y.const == 0:
                rep = y
            elif y.const == 1:
                rep = x
            elif y.const == -1:
                rep = negative(x)
        if rep is not None:
            self.add_rep(op, rep)

    @visit.on_type(add)
    def visit(self, op):
        x, y = op.args
        rep = None
        if isinstance(x, Constant):
            if x.const == 0:
                rep = y
        elif isinstance(y, Constant):
            if y.const == 0:
                rep = x
        if rep is not None:
            self.add_rep(op, rep)

    @visit.on_type(sum)
    def visit(self, op):
        x, = op.args
        if isinstance(x, Constant):
            val = x.const * op.reduction_axes.size
            self.add_rep(op, Constant(val))

    @visit.on_type(log)
    def visit(self, op):
        x, = op.args
        if isinstance(x, divide):
            num, denom = x.args
            if isinstance(num, exp):
                exp_x, = num.args
                self.add_rep(op, exp_x - type(op)(denom))
        elif isinstance(x, exp):
            exp_x, = x.args
            self.add_rep(op, exp_x)
