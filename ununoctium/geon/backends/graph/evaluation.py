import numpy as np
import numbers
import geon.backends.graph.graph as graph
from geon.backends.graph.names import axes_shape, axes_reshape
import pycuda.gpuarray as gpuarray
import pycuda.cumath as cumath
import geon.backends.graph.cudagpu as cudagpu

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

    def array_as_axes(self, axes):
        return maybe_reshape(self.array, axes_reshape(self.axes, axes))

    def __repr__(self):
        return '{array}:{axes}'.format(axes=self.axes, array=self.array)


class Environment(dict):
    def __init__(self, graph, **kvargs):
        super(Environment, self).__init__(**kvargs)
        self.graph = graph
        self.ops = graph.ordered_ops
        self.parent = None

    def child(self, **kvargs):
        result = self.__class__(graph=self.graph, **kvargs)
        result.parent = self
        return result

    def evaluate(self, result, **kvargs):
        env = self.child(**kvargs)
        vals = {}
        for op in self.ops:
            args = [vals[arg.output] for arg in op.inputs]
            if op.output is op:
                val = op.evaluate(env, *args)
                vals[op] = val
            else:
                val = op.evaluate(env, vals[op.output], *args)
            if op in result:
                env[op] = val
        return [env[op] for op in result]


    def __getitem__(self, item):
        try:
            return super(Environment, self).__getitem__(item)
        except KeyError as e:
            if self.parent is not None:
                return self.parent[item]
            raise e

    def set_vars(self, **kvargs):
        for var, value in kvargs:
            self.set_var(var, value)


class NumPyEnvironment(Environment):
    def __init__(self, **kargs):
        super(NumPyEnvironment, self).__init__(**kargs)

    def constant(self, value):
        return ArrayWithAxes(value, ())

    def input(self, name, graph_type):
        value = self[name]
        if graph_type.axes != value.axes:
            raise graph.IncompatibleShapesError()
        return value

    def absolute(self, x, out=None):
        return ArrayWithAxes(np.abs(x.array_as_axes(out.axes), out=out.array), out.axes)

    def add(self, x, y, out=None):
        return ArrayWithAxes(np.add(x.array_as_axes(out.axes), y.array_as_axes(out.axes), out=out.array), out.axes)

    def cos(self, x, out=None):
        return np.cos(x, out=out)

    def dot(self, x, y, out=None):
        return np.dot(x, y, out=out)

    def empty(self, axes, dtype):
        return ArrayWithAxes(np.empty(axes_shape(axes), dtype), axes)

    def exp(self, x, out=None):
        return ArrayWithAxes(np.exp(x.array_as_axes(out.axes), out=out.array), out.axes)

    def log(self, x, out=None):
        return ArrayWithAxes(np.log(x.array_as_axes(out.axes), out=out.array), out.axes)

    def maximum(self, x, y, out=None):
        return np.maximum(x, y, out=out)

    def minimum(self, x, y, out=None):
        return np.minimum(x, y, out=out)

    def multiply(self, x, y, out=None):
        return ArrayWithAxes(np.multiply(x.array_as_axes(out.axes), y.array_as_axes(out.axes), out=out.array), out.axes)

    def negative(self, x, out=None):
        return ArrayWithAxes(np.negative(x.array_as_axes(out.axes), out=out.array), out.axes)

    def ones(self, axes, dtype):
        return ArrayWithAxes(np.ones(axes_shape(axes), dtype), axes)

    def reciprocal(self, x, out=None):
        return ArrayWithAxes(np.reciprocal(x.array_as_axes(out.axes), out=out.array), out.axes)

    def reshape(self, x, shape):
        return x.reshape(shape)

    def sig(self, x, out):
        xa = x.array_as_axes(out.axes)
        np.negative(xa, out.array)
        np.exp(out.array, out.array)
        np.add(out.array, 1.0, out.array)
        return ArrayWithAxes(np.reciprocal(out.array, out.array), out.axes)

    def sign(self, x, out=None):
        return np.sign(x, out=out)

    def sin(self, x, out=None):
        return np.sin(x, out=out)

    def sqrt(self, x, out=None):
        return np.sqrt(x, out=out)

    def square(self, x, out=None):
        return np.square(x, out=out)

    def subtract(self, x, y, out=None):
        return np.subtract(x, y, out=out)

    def tanh(self, x, out=None):
        return np.tanh(x, out=out)

    def transpose(self, x):
        return x.transpose()

    def zeros(self, axes, dtype):
        return ArrayWithAxes(np.zeros(axes_shape(axes), dtype), axes)


class PyCUDAEnvironment(Environment):
    """
    Uses PuCUDA to evaluate.  Not fully tested; PyCUDA does not expose all the NumPy API.
    """
    def __init__(self, **kvargs):
        super(PyCUDAEnvironment, self).__init__(**kvargs)

    def evaluate(self, result, **kvargs):
        with cudagpu.cuda_device_context():
            return super(PyCUDAEnvironment, self).evaluate(result, **kvargs)

    def constant(self, value):
        return value

    def input(self, name, graph_type):
        value = gpuarray.to_gpu(self[name])
        if graph_type.shape != value.shape:
            raise graph.IncompatibleShapesError()
        return value

    def absolute(self, x, out=None):
        cumath.fabs(x, out=out)
        return out

    def add(self, x, y, out=None):
        x._axpbyz(1, y, 1, out)
        return out

    def cos(self, x, out=None):
        cumath.cos(x, out=out)
        return out

    def dot(self, x, y, out=None):
        cumath.dot(x,y, out=out)
        return out

    def empty(self, axes, dtype):
        return ArrayWithAxes(gpuarray.empty(axes_shape(axes), dtype), axes)

    def exp(self, x, out=None):
        cumath.exp(x, out=out)
        return out

    def log(self, x, out=None):
        cumath.log(x, out=out)
        return out

    def maximum(self, x, y, out=None):
        cumath.maximum(x, y, out=out)
        return out

    def minimum(self, x, y, out=None):
        cumath.minimum(x, y, out=out)
        return out

    def multiply(self, x, y, out=None):
        if isinstance(x, gpuarray.GPUArray):
            if isinstance(y, gpuarray.GPUArray):
                x._elwise_multiply(y, out=out)
                return out
            x._axpbz(y, 0, out)
        elif isinstance(y, gpuarray.GPUArray):
            y._axpbz(x, 0, out)
            return out
        else:
            return x*y

    def negative(self, x, out=None):
        x._axpbz(-1, 0.0, out)
        return out

    def ones(self, axes, dtype):
        result = gpuarray.empty(axes_shape(axes), dtype)
        result.fill(1.0)
        return ArrayWithAxes(result, axes)

    def reciprocal(self, x, out=None):
        x._rdiv_scalar(1.0, out)
        return out

    def reshape(self, x, shape):
        return x.reshape(shape)

    def sig(self, x, out):
        self.negative(x, out=out)
        cumath.exp(out, out=out)
        # Add one
        out._axpbz(1.0, 1.0, out=out)
        out._rdiv_scalar(1.0, out=out)
        return out

    def sign(self, x, out):
        out.set(np.sign(x.get()))
        return out

    def sin(self, x, out=None):
        cumath.sin(x, out=out)
        return out

    def sqrt(self, x, out=None):
        cumath.sqrt(x, out=out)
        return out

    def square(self, x, out=None):
        return self.multiply(x, x, out)

    def subtract(self, x, y, out=None):
        x._axpbyz(1, y, 1, out)
        return out

    def tanh(self, x, out=None):
        cumath.tanh(x, out=out)
        return out

    def transpose(self, x):
        return x.transpose()

    def zeros(self, axes, dtype):
        return ArrayWithAxes(gpuarray.zeros(axes_shape(axes), dtype), axes)


class GenNumPy(Environment):

    def __init__(self, **kvargs):
        super(GenNumPy, self).__init__(**kvargs)

    def evaluate(self, result, **kvargs):
        liveness = self.graph.analyze_liveness(result)

        def varname(op):
            return 't%d' % (op.opid)

        env = self.child(**kvargs)
        body = []
        for i, op in enumerate(self.ops):
            live = [varname(l) for l in liveness[i]]
            args = [varname(arg.output) for arg in op.inputs]
            if op.output is op:
                val = op.evaluate(env, *args)
                body.append('{var} = {val} # Live={live}'.format(var=varname(op), val=val, live=live))
            else:
                val = '{val} # Live={live}'.format(val=op.evaluate(env, varname(op.output), *args), live=live)
                body.append(val)
            if op in result:
                env[op] = val
        for line in body:
            print(line)
        return [env[op] for op in result]

    def constant(self, value):
        return value

    def input(self, name, graph_type):
        return 'input("{name}")'.format(name=name)

    def absolute(self, x, out=None):
        return 'np.abs({x}, out={out})'.format(x=x, out=out)

    def add(self, x, y, out=None):
        return 'np.add({x}, {y}, out={out})'.format(x=x, y=y, out=out)

    def cos(self, x, out=None):
        return 'np.cos({x}, out={out})'.format(x=x, out=out)

    def dot(self, x, y, out=None):
        return 'np.dot({x}, {y}, out={out})'.format(x=x, y=y, out=out)

    def empty(self, axes, dtype):
        return 'np.empty({axes}, np.{dtype})'.format(axes=axes, dtype=dtype)

    def exp(self, x, out=None):
        return 'np.exp({x}, out={out})'.format(x=x, out=out)

    def log(self, x, out=None):
        return 'np.log({x}, out={out})'.format(x=x, out=out)

    def maximum(self, x, y, out=None):
        return 'np.maximum({x}, {y}, out={out})'.format(x=x, y=y, out=out)

    def minimum(self, x, y, out=None):
        return 'np.minimum({x}, {y}, out={out})'.format(x=x, y=y, out=out)

    def multiply(self, x, y, out=None):
        return 'np.multiply({x}, {y}, out={out})'.format(x=x, y=y, out=out)

    def negative(self, x, out=None):
        return 'np.negative({x}, out={out})'.format(x=x, out=out)

    def ones(self, axes, dtype):
        return 'np.ones({axes}, np.{dtype})'.format(axes=axes, dtype=dtype)

    def reciprocal(self, x, out=None):
        return 'np.reciprocal({x}, out={out})'.format(x=x, out=out)

    def reshape(self, x, shape):
        return '{x}.reshape({shape})'.format(x=x, shape=shape)

    def sig(self, x, out=None):
        return 'np.negative({x}, {out})\nnp.exp({out}, {out})\nnp.add({out}, 1.0, {out}nnp.reciprocal({out}, {out})'.format(x=x, out=out)

    def sign(self, x, out=None):
        return 'np.sign({x}, out={out})'.format(x=x, out=out)

    def sin(self, x, out=None):
        return 'np.sin({x}, out={out})'.format(x=x, out=out)

    def sqrt(self, x, out=None):
        return 'np.sqrt({x}, out={out})'.format(x=x, out=out)

    def square(self, x, out=None):
        return 'np.square({x}, out={out})'.format(x=x, out=out)

    def subtract(self, x, y, out=None):
        return 'np.subtract({x}, {y}, out={out})'.format(x=x, y=y, out=out)

    def tanh(self, x, out=None):
        return 'np.tanh({x}, out={out})'.format(x=x, out=out)

    def transpose(self, x):
        return '{x}.transpose()'.format(x=x)

    def zeros(self, axes, dtype):
        return 'np.zeros({axes}, np.{dtype})'.format(axes=axes,dtype=dtype)


