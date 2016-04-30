from contextlib import contextmanager
import weakref
import numbers

from geon.backends.graph.names import AxisGenerator, NameableValue, VariableBlock
import geon.backends.graph.typing as typing
from geon.backends.graph.errors import *
from geon.backends.graph.environment import get_default_graph, set_default_graph, bound_graph, Environment, get_current_environment
from geon.backends.graph.names import Naming

import numpy as np


def maybe_reshape(array, shape):
    if isinstance(array, numbers.Real):
        return array
    if array.shape == shape:
        return array
    return array.reshape(shape)


class ArrayWithAxes(object):
    def __init__(self, array, axes):
        self.array = array
        self.axes = axes
        environment = get_current_environment()
        for axis, length in zip(axes, array.shape):
            axis[length]

    def array_as_axes(self, axes):
        return maybe_reshape(self.array, axes_reshape(self.axes, axes))

    def __repr__(self):
        return '{array}:{axes}'.format(axes=self.axes, array=self.array)


class GraphMetaclass(type):
    """Ensures that there is a default graph while running __init__"""
    def __new__(cls, name, parents, attrs):
        return super(GraphMetaclass, cls).__new__(cls, name, parents, attrs)

    def __call__(cls, *args, **kargs):
        with bound_graph():
            return super(GraphMetaclass, cls).__call__(*args, **kargs)


class GraphComponent(object):
    """
    Superclass for all models.

    Ensures that __metaclass__ is set.
    """
    __metaclass__ = GraphMetaclass

    def __init__(self, **kargs):
        super(GraphComponent, self).__init__(**kargs)
        self.naming = Naming(name="graph")
        self.environment = Environment()
        set_default_graph(self)
        self.a = AxisGenerator('a')
        self.root_context = ControlBlock()
        self.context = self.root_context
        self.inputs = {}
        self.variables = VariableBlock()
        self.op_adjoints = {}
        self.ordered_ops = []

    @contextmanager
    def bound_context(self, context):
        old_context = self.context
        try:
            self.context = context
            yield ()
        finally:
            self.context = old_context

    def add_op(self, op):
        opid = np.int64(len(self.ordered_ops))
        self.ordered_ops.append(op)
        self.context.add_context_op(op)
        return opid

    @staticmethod
    def get_ordered_ops(op, ordered_ops):
        """
        Get dependent ops ordered for autodiff.
        """
        if op not in ordered_ops:
            if isinstance(op, ArgsOp):
                for arg in op.inputs:
                    GraphComponent.get_ordered_ops(arg, ordered_ops)
            ordered_ops.append(op)

    def analyze_liveness(self, results):
        liveness = [set() for op in self.ordered_ops]
        i = len(liveness) - 1
        for result in results:
            liveness[i].add(result.output)
        while i > 0:
            op = self.ordered_ops[i]
            prealive = liveness[i - 1]
            alive = set(liveness[i])
            if isinstance(op, ValueOp):
                output = op.output
                alive.discard(output)
                for arg in op.inputs:
                    alive.add(arg.output)
                prealive |= alive
            i = i - 1
        self.liveness = liveness
        return liveness


class Model(GraphComponent):
    def __init__(self, **kargs):
        super(Model, self).__init__(**kargs)


def posneg(x):
    s = .5*sgn(x)

    return .5+s, .5-s


def axes_reshape(in_axes, out_axes):
    """
    Compute the reshape shape to broadcase in to out.  Axes must be consistently ordered

    :param in_axes: Axes of the input
    :param out_axes: Axes of the output
    :return: shape argument for reshape()
    """
    result = []
    for out_axis in out_axes:
        if out_axis in in_axes:
            result.append(out_axis.size())
        else:
            result.append(1)
    return tuple(result)


class ControlBlock(object):

    def __init__(self, last_op=None, **kargs):
        super(ControlBlock, self).__init__(**kargs)
        # Ops in this block
        self.__ops = []
        # Predecessor of this block
        self._last_op = last_op

    def add_context_op(self, op):
        if self._last_op is not None:
            op.predecessors.add(self._last_op)
        self._last_op = op
        self.__ops.append(op)

    @property
    def ops(self):
        return self.__ops


def show_graph(g):
    ids = {}

    def opids(g):
        ids[g] = len(ids)
        for op in g.ops:
            opids(op)

    opids(g.root_context)

    def show_op(g):
        for op in g.ops:
            name = ''
            if op.name is not None:
                name = op.name
            outid = ''
            if op.output is not op:
                outid = '=>%d' % (ids[op.output],)

            print '%d:%s%s:%s%s%s' % (ids[op], name, op.graph_type.axes, op, tuple(ids[arg] for arg in op.inputs), outid)
            show_op(op)
    show_op(g.root_context)

