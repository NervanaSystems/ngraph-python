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
import weakref

import numpy as np
from builtins import object, str

from geon.backends.graph.environment import get_current_environment,\
    get_current_ops, captured_ops
from geon.op_graph.arrayaxes import get_batch_axes, TensorDescription, \
    AxisIDTuple, Axes, AxesAxis, Axis
from geon.op_graph.nodes import Node, generic_method


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

    def variables(self):
        """Return all parameters used in computing this node"""
        params = []
        visited = set()
        unvisited = [self]

        while unvisited:
            node = unvisited.pop()
            visited.add(node)
            if isinstance(node, Variable):
                params.append(node)
            unvisited.extend(node.args)

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

        if len(self.axes.value) == 0:
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

    @property
    def ops(self):
        return []

    @staticmethod
    def simple_prune(results):
        SimplePrune(results)

    @property
    def output_view_info(self):
        raise NotImplementedError

    def transform(self, transformer, *args):
        """Process op"""
        pass

    def sync(self, transformer):
        """Make sure transformer has local changes"""
        pass

    def __str__(self):
        return '<{cl}:{id}>'.format(cl=self.__class__.__name__, id=id(self))


class TensorAxesInfo(object):
    """Information about a use of a tensor with axes"""

    def __init__(
            self,
            axes,
            alloc=None,
            read_only=False,
            tags=(),
            dtype=np.float32,
            **kargs):
        super(TensorAxesInfo, self).__init__(**kargs)
        axes = Axes(*axes)
        self.axes = axes
        self.views = weakref.WeakValueDictionary()
        self.alloc = alloc
        self.read_only = read_only
        self.dtype = np.dtype(dtype)
        self.tags = set(tags)
        self.__tensor_description = None
        self.initializer = None
        self.initialized = False
        self.views[self.axes] = self

    @property
    def tensor_description(self):
        if self.__tensor_description is None:
            self.__tensor_description = TensorDescription(
                axes=self.axes, dtype=self.dtype)
        return self.__tensor_description

    @property
    def value(self):
        return self.tensor_description.value

    def set_tensor(self, transformer, tensor):
        description = self.tensor_description
        description.value = transformer.fill_tensor_in(description, tensor)
        self.update_views(transformer, True)

    def update_views(self, transformer, force):
        for view in list(self.views.values()):
            if view.tensor_description is self.tensor_description:
                continue
            view.tensor_description.buffer = self.tensor_description.buffer
            view.update_tensor(transformer, force)

    def allocate(self, transformer):
        buffer = self.tensor_description.buffer
        if buffer.data is None:
            buffer.data = transformer.make_raw_buffer(buffer.size)
        if self.alloc is not None:
            tensor = self.alloc(transformer, self.tensor_description)
        else:
            tensor = transformer.tensor_view(self.tensor_description)
        self.set_tensor(transformer, tensor)
        self.update_views(transformer, False)

    def get_or_default(self, axes, default_function):
        if axes in self.views:
            return self.views[axes]
        result = default_function()
        self.views[axes] = result
        return result

    def reaxe(self, reaxe):
        return self.get_or_default(Axes(*reaxe),
                                   lambda: TensorReaxeViewInfo(
                                       tensor_axes_info=self,
                                       reaxes=reaxe,
                                       idx=len(self.views)))

    def dot_reaxe_left(self, red_axis_ids, dummy_axis=None):
        return self.get_or_default(red_axis_ids,
                                   lambda: DotLeftViewInfo(
                                       tensor_axes_info=self,
                                       red_axis_ids=red_axis_ids,
                                       idx=len(self.views),
                                       dummy_axis=dummy_axis))

    def dot_reaxe_right(self, red_axis_ids, dummy_axis=None,
                        forward_axis_ids=None):
        return self.get_or_default(red_axis_ids,
                                   lambda: DotRightViewInfo(
                                       tensor_axes_info=self,
                                       red_axis_ids=red_axis_ids,
                                       idx=len(self.views),
                                       dummy_axis=dummy_axis,
                                       forward_axis_ids=forward_axis_ids))

    def reaxe_with_dummy_axis(self, axis, dim):
        return self.get_or_default(
            ('Dummy', dim, axis),
            lambda: DummyReaxeViewInfo(
                tensor_axes_info=self, axis=axis, dim=dim, idx=len(self.views)
            )
        )


class TensorViewInfo(object):
    """The use of a view of a tensor with axes"""

    def __init__(self, tensor_axes_info, idx, **kargs):
        super(TensorViewInfo, self).__init__(**kargs)
        self.tensor_axes_info = tensor_axes_info
        self.idx = idx

    def allocate(self, transformer):
        tensor = transformer.tensor_view(self.tensor_description)
        self.tensor_description.value = tensor

    @property
    def value(self):
        return self.tensor_description.value

    def update_tensor(self, transformer, force):
        tensor_description = self.tensor_description
        if force or tensor_description.value is None:
            tensor_description.value = transformer.tensor_view(tensor_description)


class TensorReaxeViewInfo(TensorViewInfo):
    """The use of a reaxe view of a tensor with axes"""

    def __init__(self, reaxes, **kargs):
        super(TensorReaxeViewInfo, self).__init__(**kargs)
        self.reaxes = Axes(*reaxes)
        self.__tensor_description = None

    @property
    def tensor_description(self):
        if self.__tensor_description is None:
            self.__tensor_description = self.tensor_axes_info.\
                tensor_description.reaxe(self.reaxes)
        return self.__tensor_description


class DummyReaxeViewInfo(TensorViewInfo):

    def __init__(self, axis, dim, **kargs):
        super(DummyReaxeViewInfo, self).__init__(**kargs)
        self.axis = axis
        self.dim = dim
        self.__tensor_description = None

    @property
    def tensor_description(self):
        if self.__tensor_description is None:
            self.__tensor_description = self.tensor_axes_info.\
                tensor_description.reaxe_with_dummy_axis(self.axis, self.dim)
        return self.__tensor_description


class DotLeftViewInfo(TensorViewInfo):

    def __init__(self, red_axis_ids, dummy_axis=None, **kargs):
        super(DotLeftViewInfo, self).__init__(**kargs)
        self.red_axis_ids = red_axis_ids
        self.dummy_axis = dummy_axis
        self.__tensor_description = None

    @property
    def tensor_description(self):
        if self.__tensor_description is None:
            desc = self.tensor_axes_info.tensor_description
            if self.dummy_axis is not None:
                desc = desc.reaxe_with_dummy_axis(self.dummy_axis)
            self.__tensor_description = desc.dot_reaxe_left(self.red_axis_ids)
        return self.__tensor_description


class DotRightViewInfo(TensorViewInfo):

    def __init__(self, red_axis_ids, dummy_axis=None,
                 forward_axis_ids=None, **kargs):
        super(DotRightViewInfo, self).__init__(**kargs)
        self.red_axis_ids = red_axis_ids
        self.dummy_axis = dummy_axis
        self.forward_axis_ids = forward_axis_ids
        self.__tensor_description = None

    @property
    def tensor_description(self):
        if self.__tensor_description is None:
            desc = self.tensor_axes_info.tensor_description
            if self.dummy_axis is not None:
                desc = desc.reaxe_with_dummy_axis(self.dummy_axis)
            self.__tensor_description = desc.dot_reaxe_right(
                self.red_axis_ids,
                forward_axis_ids=self.forward_axis_ids
            )
        return self.__tensor_description


class AxesComp(object):
    """A Computation for computing axes"""

    def __init__(self, axes=None, **kargs):
        super(AxesComp, self).__init__(**kargs)
        self.__axes__ = axes

    @staticmethod
    def as_axes(axes, **kargs):
        if isinstance(axes, AxesComp):
            return axes
        elif axes is None:
            return None
        else:
            return LiteralAxesComp(axes=Axes(*axes), **kargs)

    @property
    def value(self):
        if self.__axes__ is None:
            self.__axes__ = self.resolve()
        return self.__axes__

    def resolve(self):
        raise NotImplementedError()

    def __add__(self, x):
        return AxesAppendComp(self, AxesComp.as_axes(x))

    def __radd__(self, x):
        return AxesAppendComp(AxesComp.as_axes(x), self)

    def __sub__(self, x):
        return AxesSubComp(self, AxesComp.as_axes(x))

    def __rsub__(self, x):
        return AxesSubComp(AxesComp.as_axes(x), self)

    def __mul__(self, x):
        return AxesIntersectComp(self, AxesComp.as_axes(x))

    def __rmul__(self, x):
        return AxesIntersectComp(AxesComp.as_axes(x), self)


def sample_axes(x, **kargs):
    return AxesSubComp(AxesComp.as_axes(x, **kargs), get_batch_axes())


def tensor_sample_axes(x, **kargs):
    return sample_axes(x.axes, **kargs)


def tensor_batch_axes(x, **kargs):
    return batch_axes(x.axes, **kargs)


def batch_axes(x, **kargs):
    return AxesIntersectComp(AxesComp.as_axes(x, **kargs), get_batch_axes())


def linear_map_axesa(in_axes, out_axes):
    return AxesSubComp(AxesAppendComp(in_axes, out_axes),
                       AxesIntersectComp(in_axes, out_axes))


def linear_map_axes(in_axes, out_axes):
    return AxesSubComp(AxesAppendComp(out_axes, in_axes),
                       AxesIntersectComp(in_axes, out_axes))


class LiteralAxesComp(AxesComp):
    """Actual axes are provided"""

    def __init__(self, **kargs):
        super(LiteralAxesComp, self).__init__(**kargs)


class ValueAxesComp(AxesComp):
    """Determine axes from value computed by x"""

    def __init__(self, x, **kargs):
        super(ValueAxesComp, self).__init__(**kargs)
        self.x = x

    def resolve(self):
        return self.x.axes.value


class AxesSubComp(AxesComp):
    """Result will be removal of axes in y from those in x"""

    def __init__(self, x, y, **kargs):
        super(AxesSubComp, self).__init__(**kargs)
        self.x = AxesComp.as_axes(x)
        self.y = AxesComp.as_axes(y)

    def resolve(self):
        x_axes = self.x.value
        y_axes = self.y.value
        return AxisIDTuple.sub(x_axes, y_axes).as_axes()


class AxesIntersectComp(AxesComp):

    def __init__(self, x, y, **kargs):
        super(AxesIntersectComp, self).__init__(**kargs)
        self.x = AxesComp.as_axes(x)
        self.y = AxesComp.as_axes(y)

    def resolve(self):
        x_axes = self.x.value
        y_axes = self.y.value
        return AxisIDTuple.intersect(x_axes, y_axes).as_axes()


class AxesAppendComp(AxesComp):

    def __init__(self, x, y, allow_repeated=False, **kargs):
        super(AxesAppendComp, self).__init__(**kargs)
        self.x = AxesComp.as_axes(x)
        self.y = AxesComp.as_axes(y)
        self.allow_repeated = allow_repeated

    def resolve(self):
        x_axes = self.x.value
        y_axes = self.y.value
        if self.allow_repeated:
            return x_axes + y_axes
        else:
            return AxisIDTuple.append(x_axes, y_axes).as_axes()


class AxesSliceComp(AxesComp):

    def __init__(self, x, lower=0, upper=None, **kargs):
        super(AxesSliceComp, self).__init__(**kargs)
        self.x = AxesComp.as_axes(x)
        self.lower = lower
        self.upper = upper

    def resolve(self):
        x_axes = self.x.value
        if self.upper is None:
            return Axes(x_axes[self.lower:])
        else:
            return x_axes[self.lower:self.upper]


# Wrapper around a function that dynamically generates axes
class AxesFuncComp(AxesComp):

    def __init__(self, func, **kargs):
        super(AxesFuncComp, self).__init__(**kargs)
        self.func = func

    def resolve(self):
        return self.func()


class Tensor(Op):

    def __init__(self, dtype=None, axes=None, scale=None, **kwds):
        super(Tensor, self).__init__(**kwds)
        if dtype is None:
            dtype = np.dtype(np.float32)
        self.dtype = dtype
        if axes is None:
            axes = ValueAxesComp(self)
        else:
            axes = AxesComp.as_axes(axes)
        self.__axes = axes
        self.__tensor_axes_info = None
        self.__call_info = None

        # Derivative will be scaled by this
        self.scale = scale

    @property
    def output(self):
        return self

    @property
    def axes(self):
        return self.__axes

    def generate_add_delta(self, adjoints, delta):
        delta_axes = delta.axes.value
        self_axes = self.axes.value
        reduction_axes = AxisIDTuple.sub(delta_axes, self_axes).as_axes()
        if reduction_axes:
            delta = sum(delta, reduction_axes=reduction_axes)

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

    @property
    def value(self):
        return self.tensor_axes_info.tensor_description.value

    @property
    def tensor_axes_info(self):
        if self.__tensor_axes_info is None:
            self.__tensor_axes_info = self.compute_tensor_axes_info()
        return self.__tensor_axes_info

    def compute_tensor_axes_info(self):
        dtype = np.float32
        if self.dtype is not None:
            dtype = self.dtype
        return TensorAxesInfo(self.axes.value, dtype=dtype, tags=self.tags)

    @property
    def output_view_info(self):
        return self.tensor_axes_info.reaxe(self.axes.value)

    @property
    def call_info(self):
        if self.__call_info is None:
            self.__call_info = self.compute_call_info()
        return self.__call_info

    def compute_call_info(self):
        return [self.reaxe(self.axes.value)]

    def transform_call_info(self, transformer, *args):
        call_args = [arg.tensor_description.value for arg in args]
        self.transform(transformer, *call_args)

    @property
    def resolved_axes(self):
        return self.tensor_axes_info.axes

    def reaxe(self, reaxe):
        return self.tensor_axes_info.reaxe(reaxe)

    def dot_reaxe_left(self, red_axis_ids, dummy_axis=None):
        return self.tensor_axes_info.dot_reaxe_left(
            red_axis_ids,
            dummy_axis=dummy_axis
        )

    def dot_reaxe_right(self, red_axis_ids, dummy_axis=None,
                        forward_axis_ids=None):
        return self.tensor_axes_info.dot_reaxe_right(
            red_axis_ids,
            dummy_axis=dummy_axis,
            forward_axis_ids=forward_axis_ids
        )

    def reaxe_with_dummy_axis(self, axis, dim):
        return self.tensor_axes_info.reaxe_with_dummy_axis(
            axis=axis,
            dim=dim
        )

    # Required for parameter initializers
    @property
    def shape(self):
        return self.__axes__()

    def mean(self, out_axes=(), **kargs):
        return mean(self, out_axes=out_axes, **kargs)


class Broadcast(Tensor):
    """
    Used to add additional axes for a returned derivative.

    """

    def __init__(self, x, **kargs):
        super(Broadcast, self).__init__(args=(x,), **kargs)

    def compute_tensor_axes_info(self):
        x, = self.args
        return x.tensor_axes_info


class ExpandDims(Tensor):

    def __init__(self, x, axis, dim, **kargs):
        self.axis = axis
        self.dim = dim
        super(ExpandDims, self).__init__(args=(x,), **kargs)

    def compute_tensor_axes_info(self):
        x, = self.args
        return x.tensor_axes_info

    def compute_call_info(self):
        return [self.output_view_info]

    @property
    def output_view_info(self):
        x, = self.args
        return x.reaxe_with_dummy_axis(self.axis, self.dim)

    def generate_adjoints(self, adjoints, delta, x):
        x.generate_add_delta(adjoints, delta)

    @property
    def axes(self):
        x_axes, dim, axis = self.args[0].axes, self.dim, self.axis

        def func():
            return x_axes.value[:dim] + Axes(axis,) + x_axes.value[dim:]
        return AxesFuncComp(func)


class AllocationOp(Tensor):

    def __init__(
            self,
            init=None,
            initial_value=None,
            **kargs):
        super(AllocationOp, self).__init__(**kargs)
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

    def __init__(self, out=None, dtype=np.float32, batch_axes=None, **kargs):
        super(ComputationOp, self).__init__(**kargs)
        self.dtype = dtype

        for arg in self.args:
            arg.users.add(self)

        if batch_axes is None:
            batch_axes = get_batch_axes()

        self.batch_axes = AxesComp.as_axes(batch_axes)

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

        def allocator(transformer, tensor_description):
            return transformer.rng_normal_tensor(
                self.rng, tensor_description, loc, scale)

        self.tensor_axes_info.alloc = allocator


class Uniform(RNGOp):

    def __init__(self, low=0.0, high=1.0, size=None, **kargs):
        super(Uniform, self).__init__(axes=size, **kargs)
        self.low = low
        self.high = high

        def allocator(transformer, tensor_description):
            return transformer.rng_uniform_tensor(
                self.rng, tensor_description, low, high)

        self.tensor_axes_info.alloc = allocator


class VoidOp(ComputationOp):

    def __init__(self, **kargs):
        super(VoidOp, self).__init__(**kargs)
        self.__axes = AxesComp.as_axes(())

    @property
    def axes(self):
        return self.__axes

    def compute_call_info(self):
        # No out
        return []


class SetItem(VoidOp):

    def __init__(self, tensor, item, val, **kargs):
        super(SetItem, self).__init__(args=(tensor, val), out=tensor, **kargs)
        self.item = item

    def compute_call_info(self):
        tensor, val = self.args
        call_info = super(SetItem, self).compute_call_info()
        call_info.append(tensor.reaxe(tensor.axes.value))
        call_info.append(val.reaxe(tensor.axes.value))
        return call_info

    def transform(self, transformer, tensor, val):
        transformer.set_item(tensor, self.item, val)


class doall(VoidOp):

    def __init__(self, all, **kargs):
        super(doall, self).__init__(args=all, out=all[-1], **kargs)


class ElementWise(ComputationOp):

    def __init__(self, **kargs):
        super(ElementWise, self).__init__(**kargs)

    @property
    def axes(self):
        inputs = self.args
        result = self.args[0].axes
        for input in inputs[1:]:
            result = AxesAppendComp(result, input.axes)
        return result

    def compute_call_info(self):
        ci = super(ElementWise, self).compute_call_info()
        for arg in self.args:
            ci.append(arg.reaxe(self.axes.value))
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
        self.__axes = ValueAxesComp(self)

    def __axes__(self):
        return self.__axes

    def generate_adjoints(self, tape, delta):
        pass

    @property
    def value(self):
        return get_current_environment()[self]

    @value.setter
    def value(self, value):
        get_current_environment()[self] = value

    def sync(self, transformer):
        value = self.value
        if isinstance(value, numbers.Real):
            transformer.fill(
                self.tensor_axes_info.tensor_description.value, value)
        else:
            transformer.set_value(self, value)


class Fill(VoidOp):

    def __init__(self, tensor, const, **kargs):
        super(Fill, self).__init__(args=(tensor,), **kargs)
        self.const = const

    def compute_call_info(self):
        tensor, = self.args
        call_info = super(Fill, self).compute_call_info()
        call_info.append(tensor.reaxe(tensor.axes.value))
        return call_info

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

    def generate_adjoints(self, adjoints, delta):
        pass

    @property
    def graph_label(self):
        shapes = self.tensor_axes_info.tensor_description.shape
        if not shapes or max(shapes) <= 2:
            return str(self.const)
        if self.name == self.id:
            return 'Constant'
        return self.name

    @property
    def axes(self):
        return AxesComp.as_axes((()))

    def __str__(self):
        return '<{cl} ({const})>'.format(
            cl=self.__class__.__name__, const=self.const)


class NumPyTensor(AllocationOp):
    """
    A NumPy tensor with attached axes information
    """

    def __init__(self, nptensor, **kargs):
        self.nptensor = nptensor
        super(NumPyTensor, self).__init__(dtype=nptensor.dtype, **kargs)

        def allocator(transformer, tensor_description):
            return transformer.nparray(tensor_description, nptensor)

        self.tensor_axes_info.alloc = allocator

    @property
    def graph_label(self):
        return str(self.nptensor.shape)

    def generate_adjoints(self, adjoints, delta):
        pass

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
            max_axes = tensor_sample_axes(x)
        self.max_axes = AxesComp.as_axes(max_axes)
        super(argmax, self).__init__(args=(x,), dtype=np.int64, **kargs)

    def compute_call_info(self):
        x, = self.args
        return [self.reaxe([self.axes.value]), x.reaxe(
            [self.max_axes.value, self.axes.value])]

    def transform(self, transformer, out, x):
        transformer.argmax(x, out)

    @property
    def axes(self):
        x, = self.args
        return AxesSubComp(x.axes, self.max_axes)


class argmin(ComputationOp):

    def __init__(self, x, min_axes=None, **kargs):
        if min_axes is None:
            min_axes = tensor_sample_axes
        self.min_axes = AxesComp.as_axes(min_axes)
        super(argmin, self).__init__(args=(x,), dtype=np.int64, **kargs)

    def compute_call_info(self):
        x, = self.args
        return [self.reaxe([self.axes.value]), x.reaxe(
            [self.min_axes.value, self.axes.value])]

    def transform(self, transformer, out, x):
        transformer.argmin(x, out)

    @property
    def axes(self):
        x, = self.args
        return AxesSubComp(x.axes, self.min_axes)


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
        self.__axis_id_info = None
        self.use_numpy_matching = numpy_matching
        self.reduction_axes = reduction_axes
        self.out_axes = out_axes
        self.forward_dot = forward_dot
        super(dot, self).__init__(args=(x, y), **kargs)

    @property
    def axis_id_info(self):
        if self.__axis_id_info is None:
            dummy = None
            x, y = self.args
            x_axes = x.axes.value
            y_axes = y.axes.value

            x_axis_ids = x_axes.as_axis_ids()
            y_axis_ids = y_axes.as_axis_ids()

            if self.forward_dot is not None:
                y_axis_ids = self.forward_dot.axis_id_info[0]

            if self.use_numpy_matching:
                out_axis_ids = x_axis_ids[:-1]\
                    + y_axis_ids[:-2]\
                    + AxisIDTuple(y_axis_ids[-1],)
                red_axis_ids = AxisIDTuple(y_axis_ids[-1],)
            else:
                if self.reduction_axes is None:
                    red_axis_ids = AxisIDTuple.intersect(
                        x_axis_ids,
                        y_axis_ids
                    )
                else:
                    red_axis_ids = self.reduction_axes.value.as_axis_ids()

                if self.out_axes is not None:
                    out_axis_ids = self.out_axes.value.as_axis_ids()
                else:
                    out_axis_ids = (
                        (x_axis_ids - red_axis_ids) +
                        (y_axis_ids - red_axis_ids)
                    )
                red_axis_ids -= out_axis_ids

                if len(red_axis_ids) == 0:
                    dummy = Axis(1)
                    red_axis_ids = AxisIDTuple(dummy[0],)

            self.__axis_id_info = (out_axis_ids, red_axis_ids, dummy)
        return self.__axis_id_info

    def compute_call_info(self):
        x, y = self.args
        out_axis_ids, red_axis_ids, dummy = self.axis_id_info
        if self.forward_dot is None:
            forward_axis_ids = None
        else:
            forward_axis_ids = self.forward_dot.axis_id_info[0]

        a = x.dot_reaxe_left(red_axis_ids, dummy_axis=dummy)
        b = y.dot_reaxe_right(
            red_axis_ids,
            forward_axis_ids=forward_axis_ids,
            dummy_axis=dummy
        )
        a_axes, b_axes = a.tensor_description.axes,\
            b.tensor_description.axes
        o = self.reaxe(a_axes[:-1] + b_axes[1:])
        return [o, a, b]

    def transform(self, transformer, out, x, y):
        transformer.dot(x, y, out)

    @property
    def axes(self):
        if self.out_axes:
            return self.out_axes
        else:
            return AxesFuncComp(
                lambda dot_obj=self: dot_obj.axis_id_info[0].as_axes()
            )

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
        zs = sum(z, reduction_axes=AxesSubComp(self.x.axes, op.batch_axes))
        self.x.generate_add_delta(adjoints, (z - zs * op))


def softmax(x, softmax_axes=None, **kargs):
    if softmax_axes is None:
        softmax_axes = tensor_sample_axes(x, **kargs)
    x = x - max(x, reduction_axes=softmax_axes)
    exps = exp(x)
    Z = sum(exps, reduction_axes=softmax_axes)
    result = exps / Z
    result.add_schema(Softmax(x=x, exps=exps, Z=Z))
    return result


class ReductionOp(ComputationOp):

    def __init__(self, x, reduction_axes=None, out_axes=None, **kargs):
        self.out_axes = AxesComp.as_axes(out_axes)
        if reduction_axes is None:
            if out_axes is None:
                self.reduction_axes = sample_axes(x.axes)
            else:
                self.reduction_axes = AxesSubComp(x.axes, self.out_axes)
        else:
            self.reduction_axes = AxesComp.as_axes(reduction_axes)
        super(ReductionOp, self).__init__(args=(x,), **kargs)
        self.mode = None

    def compute_call_info(self):
        x, = self.args
        reduction_axes = self.reduction_axes.value

        if len(reduction_axes) == 0:
            # TODO do this as a reaxe to 1d or something
            xr = x.reaxe(self.axes.value)
            self.mode = 'copy'
            return [self.reaxe(self.axes.value), xr]
        else:
            np_out_axes = self.axes.value
            red_axes = [AxesAxis(reduction_axes)]
            red_axes.extend(np_out_axes)
            red_axes = Axes(*red_axes)
            self.mode = 0
            return [self.reaxe(np_out_axes), x.reaxe(red_axes)]

    @property
    def axes(self):
        if self.out_axes is not None:
            return self.out_axes
        return AxesSubComp(self.args[0].axes, self.reduction_axes)


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
        self.out_axes = AxesComp.as_axes(out_axes)
        if reduction_axes is None:
            if out_axes is None:
                self.reduction_axes = sample_axes(x.axes)
            else:
                self.reduction_axes = AxesSubComp(x.axes, self.out_axes)
        else:
            self.reduction_axes = AxesComp.as_axes(reduction_axes)
        super(tensor_size, self).__init__(args=(x,), **kargs)

    def transform(self, transformer, out):
        resolved_reduction_axes = self.reduction_axes.value
        size = resolved_reduction_axes.size
        transformer.fill(out, size)

    @property
    def axes(self):
        return AxesComp.as_axes(())

    def generate_adjoints(self, adjoints, delta, x):
        pass


class Slice(ComputationOp):

    def __init__(self, slices, x, **kargs):
        super(Slice, self).__init__(args=(x,), **kargs)
        self.slices = slices


class Pad(ComputationOp):

    def __init__(self, axes, slice, x, **kargs):
        super(Pad, self).__init__(args=(x,), **kargs)
        self._axes = axes
        self.slice = slice

    @property
    def axes(self):
        return self._axes

    def transform(self, transformer, out, x):
        transformer.pad(x, self.slice, out)

    def generate_adjoints(self, adjoints, delta, x):
        pass


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
        
    def generate_adjoints(self, adjoints, delta):
        pass


class Temporary(AllocationOp):

    def __init__(self, **kargs):
        super(Temporary, self).__init__(tags=['temp'], **kargs)

    def generate_adjoints(self, adjoints, delta):
        pass


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
            axis = AxesSubComp(axes, x.axes)
        else:
            if axes is None:
                x_sample = sample_axes(x.axes)
                x_batch = batch_axes(x.axes)
                axes = AxesAppendComp(Axes(axis), AxesAppendComp(x_sample, x_batch))
        super(onehot, self).__init__(args=(x,), axes=axes, **kargs)
        self.axis = axis

    def compute_call_info(self):
        x, = self.args
        ci = [self.reaxe(Axes(Axes(self.axis),
                              AxesSubComp(self.axes, Axes(self.axis)).value)),
              x.reaxe(Axes(x.axes.value))]
        return ci

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
    if adjoint.axes.value == indep.axes.value:
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


def cross_entropy_multi(y, t, usebits=False, out_axes=None, enable_softmax_opt=True,
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


def cross_entropy_binary_inner(y, t, enable_sig_opt=True, enable_diff_opt=True, **kargs):
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
