import numpy as np

from geon.backends.graph.errors import IncompatibleShapesError
from geon.backends.graph.arrayaxes import axes_sub
import pycuda.gpuarray as gpuarray
import pycuda.cumath as cumath
import geon.backends.graph.cudagpu as cudagpu
import geon.backends.graph.ast as ast
from geon.backends.graph.arrayaxes import find_axes_in_axes, AxisArray
import geon.backends.graph.arrayaxes as arrayaxes
from geon.backends.graph.environment import get_current_environment, get_current_ops, captured_ops
from geon.backends.graph.environment import get_batch_axes, set_batch_axes


def axes_shape(axes):
    return tuple(axis.length for axis in axes)


class Evaluator(object):
    def __init__(self, results, error=None, initialize=False, environment=None, **kvargs):
        super(Evaluator, self).__init__(**kvargs)
        if environment is None:
            environment = get_current_environment()
        self.environment = environment
        self.results = results

        self.ops = ast.Op.ordered_ops(self.results, True)

        self.opids = dict()
        for i, op in enumerate(self.ops):
            self.opids[op] = i

    def allocate(self, ops):
        for op in ops:
            self.get_resolved_tensor_axes(op)
            self.environment[op] = op.allocate(self)

    def initialize(self):
        self.allocate(self.ops)

        initializers = []
        with captured_ops(initializers):
            for op in self.ops:
                if isinstance(op, ast.Parameter):
                    op.initializer(self, self.environment[op])
        ops = ast.Op.ordered_ops(initializers, True)
        self.allocate(ops)
        self.evaluate_ops(ops)

    def get_resolved_tensor_axes(self, tensor):
        try:
            return self.environment.get_cached_resolved_tensor_axes(tensor)
        except KeyError:
            axes = tensor.axes.evaluate(self)
            self.environment.set_cached_resolved_tensor_axes(tensor, axes)
            return axes

    def get_resolved_axes(self, axes_comp):
        try:
            return self.environment[axes_comp]
        except KeyError:
            resolved_axes = axes_comp.evaluate(self)
            self.environment[axes_comp] = resolved_axes
            return resolved_axes

    def get_cached_resolved_tensor_axes(self, tensor):
        return self.environment.get_cached_resolved_tensor_axes(tensor)

    def get_batch_axes(self):
        return get_batch_axes()

    def set_batch_axes(self, axes):
        set_batch_axes(axes)

    def evaluate_ops(self, ops):
        vals = {}
        for op in ops:
            args = [vals[arg.output] for arg in op.inputs]
            if op.output is op:
                val = op.evaluate(self, *args)
                vals[op.output] = val
            else:
                val = op.evaluate(self, vals[op.output], *args)
                vals[op.output] = val
        return vals

    def evaluate(self):
        vals = self.evaluate_ops(self.ops)
        r = {}
        for op in self.results:
            r[op] = vals[op.output]
        return r


class NumPyEvaluator(Evaluator):
    def __init__(self, **kargs):
        super(NumPyEvaluator, self).__init__(**kargs)

    def trace(self, x, label, out):
        oa = out
        xa = x
        if oa.shape == ():
            oa = oa.reshape((1,))
            xa = xa.reshape((1,))
        oa[:] = xa
        return out

    def rng(self, seed=None):
        return np.random.RandomState(seed=seed)

    def rng_uniform(self, rng, low, high, out):
        shape = [axis.length for axis in out.axes]
        out[:] = rng.uniform(low, high, shape)
        return out

    def set_item(self, array, item, value):
        array.__setitem__(item, value)

    def constant(self, value, axes, dtype):
        return AxisArray(axes=axes, array=value, dtype=dtype)

    def absolute(self, x, out):
        return np.abs(x.axes_like(out), out=out)

    def add(self, x, y, out):
        return np.add(x.axes_like(out), y.axes_like(out), out=out)

    def cos(self, x, out):
        return np.cos(x.axes_like(out), out=out)

    def dot(self, x, y, red_axes, out):
        # This implementation requires axes
        #   x = xl red xr
        #   y = yl red yr
        #   out = xl xr yl yr
        #   At least one of xl, xr, yl, yr is empty

        x_axes = x.axes
        y_axes = y.axes
        out_axes = out.axes

        if len(x_axes) is 0 or len(y_axes) is 0:
            # TODO turn this into multiply ahead of time
            np.multiply(x, y, out=out)
            return out

        xi = find_axes_in_axes(red_axes, x_axes)
        if xi == -1:
            raise IncompatibleShapesError()
        yi = find_axes_in_axes(red_axes, y_axes)
        if yi == -1:
            raise IncompatibleShapesError()

        xl = x_axes[0:xi]
        xr = x_axes[xi+len(red_axes):]
        yl = y_axes[0:yi]
        yr = y_axes[yi+len(red_axes):]

        al = arrayaxes.axes_append(xl, xr)
        br = arrayaxes.axes_append(yl, yr)

        a = arrayaxes.reaxe(x, axes=(al, red_axes))
        b = arrayaxes.reaxe(y, axes=(red_axes, br))
        if arrayaxes.axes_intersect(al,br):
            # Can't handle yet
            raise IncompatibleShapesError()
        o = arrayaxes.reaxe(out, axes=(al, br))
        #o = out.reaxe(axes=(al, br))
        np.dot(a,b,out=o)
        return out

    def update(self, params, delta):
        if params.shape != delta.shape:
            print('mismatch', params.axes, delta.axes)
        np.subtract(params, delta.axes_like(params), out=params)
        return params

    def empty(self, axes, dtype):
        return AxisArray(axes=axes, dtype=dtype or np.float32)

    def exp(self, x, out):
        return np.exp(x.axes_like(out), out=out)

    def log(self, x, out):
        return np.log(x.axes_like(out), out=out)

    def maximum(self, x, y, out):
        return np.maximum(x.axes_like(out), y.axes_like(out), out=out)

    def minimum(self, x, y, out):
        return np.minimum(x.axes_like(out), y.axes_like(out), out=out)

    def multiply(self, x, y, out):
        return np.multiply(x.axes_like(out), y.axes_like(out), out=out)

    def negative(self, x, out):
        return np.negative(x.axes_like(out), out=out)

    def ones(self, axes, dtype):
        return AxisArray(axes=axes, dtype=dtype, array=np.ones(axes_shape(axes)))

    def reciprocal(self, x, out):
        return np.reciprocal(x.axes_like(out), out=out)

    def sig(self, x, out):
        xa = x.axes_like(out)
        np.negative(xa, out)
        np.exp(out, out)
        np.add(out, 1.0, out)
        return np.reciprocal(out, out)

    def sign(self, x, out):
        return np.sign(x.axes_like(out), out=out)

    def sin(self, x, out):
        return np.sin(x.axes_like(out), out=out)

    def softmax(self, x, batch_axes, out):
        softmax_axes = axes_sub(x.axes, batch_axes)
        if softmax_axes == ():
            raise ValueError('Empty softmax')
        sa_i = find_axes_in_axes(softmax_axes, x.axes)
        if sa_i == -1:
            raise ValueError('Softmax axes not contiguous')
        if sa_i != 0:
            raise ValueError('Softmax axes not on left')
        sm_dims = [axis.length for axis in softmax_axes]
        def prod(dims):
            result = 1
            for dim in dims:
                result = result * dim
            return result
        sm_size = prod(sm_dims)
        rem_dims = [axis.length for axis in x.axes[len(softmax_axes):]]

        if len(softmax_axes) > 1:
            new_shape = [sm_size]+rem_dims
            x = x.reshape(new_shape)
        m = x.max(axis=0)
        m = m.reshape([1]*len(sm_dims)+rem_dims)
        np.subtract(x, m, out=out)
        np.exp(out, out=out)
        out_temp = out.reshape([sm_size]+list(out.shape[len(softmax_axes):]))
        s = out_temp.sum(axis=0)
        s = s.reshape([1]*len(sm_dims)+list(out.shape[len(softmax_axes):]))
        return np.divide(out, s, out=out)

    def sqrt(self, x, out):
        return np.sqrt(x.axes_like(out), out=out)

    def square(self, x, out):
        return np.square(x.axes_like(out), out=out)

    def subtract(self, x, y, out):
        return np.subtract(x.axes_like(out), y.axes_like(out), out=out)

    def sum(self, x, reduction_axes, out):
        x_axes = x.axes
        np_out_axes = axes_sub(x_axes, reduction_axes)
        np_red_dims = tuple(x_axes.index(axis) for axis in reduction_axes)
        if list(out.axes) != list(np_out_axes):
            temp = np.sum(x, axis=np_red_dims)
            out[...] = temp
        else:
            np.sum(x, axis=np_red_dims, out=out.array_as_axes(np_out_axes))
        return out

    def tanh(self, x, out):
        return np.tanh(x.axes_like(out), out=out)

    def zeros(self, axes, dtype):
        return AxisArray(axes=axes, dtype=dtype, array=np.zeros(axes_shape(axes)))

    def uniform(self, x, low, high):
        u = self.rng.uniform(low, high, x.shape)
        x[:] = u


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
        return AxisArray(array=value, axes=axes, dtype=dtype)

    def absolute(self, x, out):
        cumath.fabs(x.axes_like(out), out=out)
        return out

    def add(self, x, y, out):
        x.axes_like(out)._axpbyz(1, y.axes_like(out), 1, out)
        return out

    def cos(self, x, out):
        cumath.cos(x.axes_like(out), out=out)
        return out

    def dot(self, x, y, int_axes, out):
        # TODO Implement axis dot
        cumath.dot(x, y, out=out)
        return out

    def empty(self, axes, dtype):
        return AxisArray(array=gpuarray.empty(axes_shape(axes), dtype), axes=axes)

    def exp(self, x, out):
        cumath.exp(x.axes_like(out), out=out)
        return out

    def log(self, x, out):
        cumath.log(x.axes_like(out), out=out)
        return out

    def maximum(self, x, y, out):
        cumath.maximum(x.axes_like(out), y.axes_like(out), out=out)
        return out

    def minimum(self, x, y, out):
        cumath.minimum(x.axes_like(out), y.axes_like(out), out=out)
        return out

    def multiply(self, xa, ya, out):
        x = xa.axes_like(out)
        y = ya.axes_like(out)
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

    def negative(self, x, out):
        x.axes_like(out)._axpbz(-1, 0.0, out)
        return out

    def ones(self, axes, dtype):
        result = gpuarray.empty(axes_shape(axes), dtype)
        result.fill(1.0)
        return AxisArray(array=result, axes=axes)

    def reciprocal(self, x, out):
        x.axes_like(out)._rdiv_scalar(1.0, out)
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
        out.set(np.sign(x.axes_like(out).get()))
        return out

    def sin(self, x, out):
        cumath.sin(x.axes_like(out), out=out)
        return out

    def sqrt(self, x, out):
        cumath.sqrt(x.axes_like(out), out=out)
        return out

    def square(self, x, out):
        return self.multiply(x.axes_like(out), x.axes_like(out), out)

    def subtract(self, x, y, out):
        x.axes_like(out)._axpbyz(1, y.axes_like(out), 1, out)
        return out

    def tanh(self, x, out):
        cumath.tanh(x.axes_like(out), out=out)
        return out

    def zeros(self, axes, dtype):
        return AxisArray(array=gpuarray.zeros(axes_shape(axes), dtype), axes=axes)


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

        body = []
        vals = {}
        for i, op in enumerate(self.ops):
            live = [varname(l) for l in liveness[i]]
            args = [varname(arg.output) for arg in op.inputs]
            if op.output is op:
                val = op.evaluate(self, *args)
                vals[op] = val
                body.append('{var} = {val} # Live={live}'.format(var=varname(op), val=val, live=live))
            else:
                val = '{var} = {val} # Live={live}'.format(val=op.evaluate(self, varname(op.output), *args), var=varname(op), live=live)
                vals[op] = val
                body.append(val)
        for line in body:
            print(line)
        return [vals[op] for op in self.results]

    def constant(self, value, axes, dtype):
        return 'constant {dtype} {axes} = {value}'.format(value=value, axes=axes, dtype=dtype)

    def absolute(self, x, out):
        return 'np.abs({x}, out={out})'.format(x=x, out=out)

    def add(self, x, y, out):
        return 'np.add({x}, {y}, out={out})'.format(x=x, y=y, out=out)

    def cos(self, x, out):
        return 'np.cos({x}, out={out})'.format(x=x, out=out)

    def dot(self, x, y, red_axes, out):
        return 'np.dot({x}, {y}, axes={a}, out={out})'.format(x=x, y=y, out=out, a=red_axes)

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
        return 'np.negative({x}, {out})\nnp.exp({out}, {out})\\nnp.add({out}, 1.0, {out}nnp.reciprocal({out}, {out})'.format(x=x, out=out)

    def sign(self, x, out):
        return 'np.sign({x}, out={out})'.format(x=x, out=out)

    def sin(self, x, out):
        return 'np.sin({x}, out={out})'.format(x=x, out=out)

    def softmax(self, x, batch_axes, out):
        return 'softmax({x}, batch_axes={batch_axes}, out=out'.format(x=x, batch_axes=batch_axes, out=out)

    def sqrt(self, x, out):
        return 'np.sqrt({x}, out={out})'.format(x=x, out=out)

    def square(self, x, out):
        return 'np.square({x}, out={out})'.format(x=x, out=out)

    def subtract(self, x, y, out):
        return 'np.subtract({x}, {y}, out={out})'.format(x=x, y=y, out=out)

    def sum(self, x, reduction_axes, out):
        return 'np.sum({x},axis={a}, out={out})'.format(x=x, a=reduction_axes, out=out)

    def tanh(self, x, out):
        return 'np.tanh({x}, out={out})'.format(x=x, out=out)

    def transpose(self, x):
        return '{x}.transpose()'.format(x=x)

    def zeros(self, axes, dtype):
        return 'np.zeros({axes}, np.{dtype})'.format(axes=axes,dtype=dtype)


