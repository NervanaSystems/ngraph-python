import numpy as np
import geon.backends.graph.graph as graph
import pycuda.gpuarray as gpuarray
import pycuda.cumath as cumath
import geon.backends.graph.cudagpu as cudagpu


class Environment(dict):
    def __init__(self, graph, **kvargs):
        super(Environment, self).__init__(**kvargs)
        self.graph = graph
        self.ops = graph.ordered_ops()
        self.parent = None

    def child(self, **kvargs):
        result = self.__class__(self.graph, **kvargs)
        result.parent = self
        return result

    def evaluate(self, result, **kvargs):
        env = self.child(**kvargs)
        vals = {}
        for op in self.ops:
            if not isinstance(op, graph.GraphOp):
                continue
            args = [vals[arg.out] for arg in op.args]
            if op.out is op:
                val = op.evaluate(env, *args)
                vals[op] = val
            else:
                val = op.evaluate(env, vals[op.out], *args)
            if op.name is not None:
                env[op.name] = val
        return [env[name] for name in result]


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
    def constant(self, value):
        return value

    def input(self, name, graph_type):
        value = self[name]
        if graph_type.shape != value.shape:
            raise graph.IncompatibleShapesError()
        return value

    def absolute(self, x, out=None):
        return np.abs(x, out=out)

    def add(self, x, y, out=None):
        return np.add(x, y, out=out)

    def cos(self, x, out=None):
        return np.cos(x, out=out)

    def dot(self, x, y, out=None):
        return np.dot(x, y, out=out)

    def empty(self, shape, dtype):
        return np.empty(shape, dtype)

    def exp(self, x, out=None):
        return np.exp(x, out=out)

    def log(self, x, out=None):
        return np.log(x, out=out)

    def maximum(self, x, y, out=None):
        return np.maximum(x, y, out=out)

    def minimum(self, x, y, out=None):
        return np.minimum(x, y, out=out)

    def multiply(self, x, y, out=None):
        return np.multiply(x, y, out=out)

    def negative(self, x, out=None):
        return np.negative(x, out=out)

    def ones(self, shape, dtype):
        return np.ones(shape, dtype)

    def reciprocal(self, x, out=None):
        return np.reciprocal(x, out=out)

    def reshape(self, x, shape):
        return x.reshape(shape)

    def sig(self, x, out=None):
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

    def zeros(self, shape, dtype):
        return np.zeros(shape, dtype)


class PyCUDAEnvironment(Environment):
    """
    Uses PuCUDA to evaluate.  Not fully tested; PyCUDA does not expose all the NumPy API.
    """
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

    def empty(self, shape, dtype):
        return gpuarray.empty(shape, dtype)

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

    def ones(self, shape, dtype):
        result = gpuarray.empty(shape, dtype)
        result.fill(1.0)
        return result

    def reciprocal(self, x, out=None):
        x._rdiv_scalar(1.0, out)
        return out

    def reshape(self, x, shape):
        return x.reshape(shape)

    def sig(self, x, out=None):
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

    def zeros(self, shape, dtype):
        return gpuarray.zeros(shape, dtype)


class GenNumPy(Environment):

    def evaluate(self, result, **kvargs):
        env = self.child(**kvargs)
        body = []
        vals = {}
        for i, op in enumerate(self.ops):
            var = 't%d' % (i,)
            if not isinstance(op, graph.GraphOp):
                continue
            args = [vals[arg.out] for arg in op.args]
            if op.out is op:
                val = op.evaluate(env, *args)
                body.append('{var} = {val}'.format(var=var, val=val))
            else:
                val = op.evaluate(env, vals[op.out], *args)
                body.append(val)
            vals[op] = var
            if op.name is not None:
                env[op.name] = val
        for line in body:
            print(line)
        return [env[name] for name in result]

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

    def empty(self, shape, dtype):
        return 'np.empty({shape}, np.{dtype})'.format(shape=shape, dtype=dtype)

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

    def ones(self, shape, dtype):
        return 'np.ones({shape}, np.{dtype})'.format(shape=shape, dtype=dtype)

    def reciprocal(self, x, out=None):
        return 'np.reciprocal({x}, out={out})'.format(x=x, out=out)

    def reshape(self, x, shape):
        return '{x}.reshape({shape})'.format(x=x, shape=shape)

    def sig(self, x, out=None):
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

    def zeros(self, shape, dtype):
        return 'np.zeros({shape}, np.{dtype})'.format(shape=shape,dtype=dtype)


