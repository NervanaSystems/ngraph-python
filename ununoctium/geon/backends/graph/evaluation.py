import numpy as np

from geon.backends.graph.errors import IncompatibleShapesError
from geon.backends.graph.graph import ArrayWithAxes
import pycuda.gpuarray as gpuarray
import pycuda.cumath as cumath
import geon.backends.graph.cudagpu as cudagpu
import geon.backends.graph.ast as ast

def axes_shape(axes):
    return tuple(axis.value for axis in axes)

class Evaluator(dict):
    def __init__(self, environment, results, **kvargs):
        super(Evaluator, self).__init__(**kvargs)
        self.environment = environment
        self.results = results
        self.ops = ast.Op.ordered_ops(results, True)
        self.opids = dict()
        for i, op in enumerate(self.ops):
            self.opids[op] = i
        self.parent = None

    def child(self, **kvargs):
        result = self.__class__(environment=self.environment, results=self.results, **kvargs)
        result.parent = self
        return result

    def evaluate(self, **kvargs):
        evaluator = self.child(**kvargs)
        vals = {}
        for op in self.ops:
            args = [vals[arg.output] for arg in op.inputs]
            if op.output is op:
                val = op.evaluate(evaluator, *args)
                vals[op] = val
            else:
                val = op.evaluate(evaluator, vals[op.output], *args)
            if op in self.results:
                evaluator[op] = val
        return [evaluator[op] for op in self.results]


    def __getitem__(self, item):
        try:
            return super(Evaluator, self).__getitem__(item)
        except KeyError as e:
            if self.parent is not None:
                return self.parent[item]
            raise e

    def set_input(self, input, value):
        self[input.name] = value
        self.environment.set_cached_node_axes(input, value.axes)


class NumPyEvaluator(Evaluator):
    def __init__(self, **kargs):
        super(NumPyEvaluator, self).__init__(**kargs)

    def constant(self, value, axes, dtype):
        return ArrayWithAxes(value, axes=axes, dtype=dtype)

    def input(self, name, axes, dtype):
        value = self[name]
        if axes != value.axes:
            raise IncompatibleShapesError()
        return value

    def absolute(self, x, out):
        return ArrayWithAxes(np.abs(x.array_as_axes(out.axes), out=out.array), out.axes)

    def add(self, x, y, out):
        return ArrayWithAxes(np.add(x.array_as_axes(out.axes), y.array_as_axes(out.axes), out=out.array), out.axes)

    def cos(self, x, out):
        return np.cos(x.array_as_axes(out.axes), out=out.array)

    def dot(self, x, y, red_axes, out):
        x_axes = x.axes
        y_axes = y.axes
        xi = ast.find_axes_in_axes(red_axes, x_axes)
        if xi == -1:
            raise IncompatibleShapesError()
        yi = ast.find_axes_in_axes(red_axes, y_axes)
        if yi == -1:
            raise IncompatibleShapesError()

        # Reshape x to be (leftprod, redprod)
        # Reshape y to be (leftprod, redprod, rightprod)


        
        return np.dot(x, y, out=out)

    def empty(self, axes, dtype):
        return ArrayWithAxes(np.empty(axes_shape(axes), dtype), axes)

    def exp(self, x, out):
        return ArrayWithAxes(np.exp(x.array_as_axes(out.axes), out=out.array), out.axes)

    def log(self, x, out):
        return ArrayWithAxes(np.log(x.array_as_axes(out.axes), out=out.array), out.axes)

    def maximum(self, x, y, out):
        return np.maximum(x.array_as_axes(out.shape), y.array_as_axes(out.shape), out=out.array)

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


class PyCUDAEvaluator(Evaluator):
    """
    Uses PuCUDA to evaluate.  Not fully tested; PyCUDA does not expose all the NumPy API.
    """
    def __init__(self, **kvargs):
        super(PyCUDAEvaluator, self).__init__(**kvargs)

    def evaluate(self, **kvargs):
        with cudagpu.cuda_device_context():
            return super(PyCUDAEvaluator, self).evaluate(**kvargs)

    def constant(self, value, axes, dtype):
        return ArrayWithAxes(value, axes=axes, dtype=dtype)

    def input(self, name, axes, dtype):
        value = self[name]
        if axes != value.axes:
            raise IncompatibleShapesError()

        value = gpuarray.to_gpu(value.array)
        return ArrayWithAxes(value, axes)

    def absolute(self, x, out):
        cumath.fabs(x.array_as_axes(out.axes), out=out.array)
        return out

    def add(self, x, y, out):
        x.array_as_axes(out.axes)._axpbyz(1, y.array_as_axes(out.axes), 1, out.array)
        return out

    def cos(self, x, out):
        cumath.cos(x.array_as_axes(out.axes), out=out.array)
        return out

    def dot(self, x, y, out=None):
        # TODO Implement axis dot
        cumath.dot(x.array,y.array, out=out.array)
        return out

    def empty(self, axes, dtype):
        return ArrayWithAxes(gpuarray.empty(axes_shape(axes), dtype), axes)

    def exp(self, x, out):
        cumath.exp(x.array_as_axes(out.axes), out=out.array)
        return out

    def log(self, x, out):
        cumath.log(x.array_as_axes(out.axes), out=out.array)
        return out

    def maximum(self, x, y, out):
        cumath.maximum(x.array_as_axes(out.axes), y.array_as_axes(out.axes), out=out)
        return out

    def minimum(self, x, y, out):
        cumath.minimum(x.array_as_axes(out.axes), y.array_as_axes(out.axes), out=out)
        return out

    def multiply(self, xa, ya, out):
        x = xa.array_as_axes(out.axes)
        y = ya.array_as_axes(out.axes)
        o = out.array
        if isinstance(x, gpuarray.GPUArray):
            if isinstance(y, gpuarray.GPUArray):
                x._elwise_multiply(y, out=o)
                return out
            x._axpbz(y, 0, o)
        elif isinstance(y, gpuarray.GPUArray):
            y._axpbz(x, 0, o)
            return out
        else:
            return x*y

    def negative(self, x, out):
        x.array_as_axes(out.axes)._axpbz(-1, 0.0, out.array)
        return out

    def ones(self, axes, dtype):
        result = gpuarray.empty(axes_shape(axes), dtype)
        result.fill(1.0)
        return ArrayWithAxes(result, axes)

    def reciprocal(self, x, out):
        x.array_as_axes(out.axes)._rdiv_scalar(1.0, out.array)
        return out

    def reshape(self, x, shape):
        return x.reshape(shape)

    def sig(self, x, out):
        self.negative(x, out=out)
        cumath.exp(out.array, out=out.array)
        # Add one
        out.array._axpbz(1.0, 1.0, out=out.array)
        out.array._rdiv_scalar(1.0, out=out.array)
        return out

    def sign(self, x, out):
        out.array.set(np.sign(x.array_as_axes(out.axes).get()))
        return out

    def sin(self, x, out):
        cumath.sin(x.array_as_axes(out.axes), out=out.array)
        return out

    def sqrt(self, x, out):
        cumath.sqrt(x.array_as_axes(out.axes), out=out.array)
        return out

    def square(self, x, out):
        return self.multiply(x.array_as_axes(out.axes), x.array_as_axes(out.axes), out.array)

    def subtract(self, x, y, out):
        x.array_as_axes(out.axes)._axpbyz(1, y.array_as_axes(out.axes), 1, out.array)
        return out

    def tanh(self, x, out):
        cumath.tanh(x.array_as_axes(out.axes), out=out.array)
        return out

    def transpose(self, x):
        return x.array.transpose()

    def zeros(self, axes, dtype):
        return ArrayWithAxes(gpuarray.zeros(axes_shape(axes), dtype), axes)


class GenNumPy(Evaluator):

    def __init__(self, **kvargs):
        super(GenNumPy, self).__init__(**kvargs)

    def evaluate(self, **kvargs):
        liveness = ast.Op.analyze_liveness(self.results, self.ops)

        def varname(op):
            try:
                return 't%d' % self.opids[op]
            except KeyError:
                return "Error on "+str(op)

        evaluator = self.child(**kvargs)
        body = []
        for i, op in enumerate(self.ops):
            live = [varname(l) for l in liveness[i]]
            args = [varname(arg.output) for arg in op.inputs]
            if op.output is op:
                val = op.evaluate(evaluator, *args)
                body.append('{var} = {val} # Live={live}'.format(var=varname(op), val=val, live=live))
            else:
                val = '{val} # Live={live}'.format(val=op.evaluate(evaluator, varname(op.output), *args), live=live)
                body.append(val)
            if op in self.results:
                evaluator[op] = val
        for line in body:
            print(line)
        return [evaluator[op] for op in self.results]

    def constant(self, value, axes, dtype):
        return 'constant {dtype} {axes} = {value}'.format(value=value, axes=axes, dtype=dtype)

    def input(self, name, axes, dtype):
        return 'input("{name}")'.format(name=name)

    def absolute(self, x, out):
        return 'np.abs({x}, out={out})'.format(x=x, out=out)

    def add(self, x, y, out):
        return 'np.add({x}, {y}, out={out})'.format(x=x, y=y, out=out)

    def cos(self, x, out):
        return 'np.cos({x}, out={out})'.format(x=x, out=out)

    def dot(self, x, y, out):
        return 'np.dot({x}, {y}, out={out})'.format(x=x, y=y, out=out)

    def empty(self, axes, dtype):
        return 'np.empty({axes}, np.{dtype})'.format(axes=axes, dtype=dtype)

    def exp(self, x, out):
        return 'np.exp({x}, out={out})'.format(x=x, out=out)

    def log(self, x, out):
        return 'np.log({x}, out={out})'.format(x=x, out=out)

    def maximum(self, x, y, out):
        return 'np.maximum({x}, {y}, out={out})'.format(x=x, y=y, out=out)

    def minimum(self, x, y, out):
        return 'np.minimum({x}, {y}, out={out})'.format(x=x, y=y, out=out)

    def multiply(self, x, y, out):
        return 'np.multiply({x}, {y}, out={out})'.format(x=x, y=y, out=out)

    def negative(self, x, out):
        return 'np.negative({x}, out={out})'.format(x=x, out=out)

    def ones(self, axes, dtype):
        return 'np.ones({axes}, np.{dtype})'.format(axes=axes, dtype=dtype)

    def reciprocal(self, x, out):
        return 'np.reciprocal({x}, out={out})'.format(x=x, out=out)

    def reshape(self, x, shape):
        return '{x}.reshape({shape})'.format(x=x, shape=shape)

    def sig(self, x, out):
        return 'np.negative({x}, {out})\nnp.exp({out}, {out})\nnp.add({out}, 1.0, {out}nnp.reciprocal({out}, {out})'.format(x=x, out=out)

    def sign(self, x, out):
        return 'np.sign({x}, out={out})'.format(x=x, out=out)

    def sin(self, x, out):
        return 'np.sin({x}, out={out})'.format(x=x, out=out)

    def sqrt(self, x, out):
        return 'np.sqrt({x}, out={out})'.format(x=x, out=out)

    def square(self, x, out):
        return 'np.square({x}, out={out})'.format(x=x, out=out)

    def subtract(self, x, y, out):
        return 'np.subtract({x}, {y}, out={out})'.format(x=x, y=y, out=out)

    def tanh(self, x, out):
        return 'np.tanh({x}, out={out})'.format(x=x, out=out)

    def transpose(self, x):
        return '{x}.transpose()'.format(x=x)

    def zeros(self, axes, dtype):
        return 'np.zeros({axes}, np.{dtype})'.format(axes=axes,dtype=dtype)


