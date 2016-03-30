import numpy as np

from geon.backends.graph.errors import *


class ParamTypeFactory(object):
    def __init__(self, type_constructor):
        self.type_constructor = type_constructor

    def __getitem__(self, args):
        if not isinstance(args, tuple):
            args = tuple((args,))
        return self.type_constructor(*args)


class GraphType(object):
    def is_subtype_of(self, supertype):
        return self == supertype


class ShapeType(object):
    pass


class IterableType(GraphType):
    def __init__(self, itertype):
        self.itertype = itertype

    def is_subtype_of(self, supertype):
        return isinstance(supertype, IterableType) and self.itertype.is_subtype_of(supertype.itertype)


class IteratorType(GraphType):
    def __init__(self, iterable_type):
        self.iterable_type = iterable_type


    def is_subtype_of(self, supertype):
        return isinstance(supertype, IteratorType) and self.iterabletype.is_subtype_of(supertype.iterabletype)


class ArrayType(GraphType, ShapeType):
    def __init__(self, shape, dtype=None):
        self.shape = shape
        self.dtype = np.dtype(dtype or np.float32)

    def is_subtype_of(self, supertype):
        if not isinstance(supertype, ArrayType):
            return False
        return self.shape == supertype.shape and self.dtype == supertype.dtype

    def array_args(self):
        return self.shape, self.dtype


class TupleType(GraphType, ShapeType):
    def __init__(self, *types):
        self.types = types
        self.shape = ()
        if len(types) > 1:
            first, = types
            if isinstance(first, ShapeType):
                self.shape = first.shape


class VoidType(GraphType):
    pass


Iterable = ParamTypeFactory(IterableType)
Iterator = ParamTypeFactory(IteratorType)
Array = ParamTypeFactory(ArrayType)
Tuple = ParamTypeFactory(TupleType)
Void = VoidType()

def graph_shape(graph_type):
    if isinstance(graph_type, ArrayType):
        return graph_type.shape
    return ()


def elementwise_shape(*shapes):
    n = max((len(shape) for shape in shapes))

    def prepend(s):
        return tuple(1 for x in xrange(n-len(s)))+s

    shapes = (prepend(s) for s in shapes)

    def broadcast(*vals):
        result = 1
        for val in vals:
            if val == 1:
                continue
            elif result == 1 or val == result:
                result = val
            else:
                raise IncompatibleShapesError()
        return result

    return tuple(broadcast(*d) for d in zip(*shapes))


def argument_dtype(dtype, *args):
    return np.result_type(dtype, *(arg.dtype for arg in args))


def elementwise_graph_type(dtype, *graph_types):
    shapes = (graph_shape(graph_type) for graph_type in graph_types)
    shape = elementwise_shape(*shapes)

    dtype = np.result_type(dtype, *(graph_type.dtype for graph_type in graph_types))

    return Array[shape, dtype]


def graph_type(o):
    if isinstance(o, np.ndarray):
        return Array[o.shape, o.dtype]
    if isinstance(o, tuple):
        return Tuple(*(graph_type(e) for e in o))
    return np.dtype(type(o))



