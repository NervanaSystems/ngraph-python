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
from builtins import object
from contextlib import contextmanager
from orderedset import OrderedSet

import numpy as np
from copy import deepcopy

import ngraph as ng
import ngraph.transformers as ngt


class ExecutorFactory(object):
    """TODO."""

    def __init__(self):
        pass

    def __enter__(self):
        self.transformer = ngt.make_transformer()
        if is_flex_transformer(self.transformer):
            self.cpu_transformer = ngt.Transformer.transformers['cpu']()
        return self

    def __exit__(self, *args):
        if is_flex_transformer(self.transformer):
            self.cpu_transformer.close()
        self.transformer.close()

    def executor(self, results, *parameters):
        return self.transformer.computation(results, *parameters)

    def get_tensor_view_value(self, op, host_tensor=None):
        return self.transformer.get_tensor_view_value(op, host_tensor)

    @staticmethod
    def get_all_placeholders(graph):
        placeholders = []
        ops = [graph]
        for op in ops:
            for arg in op.args:
                if isinstance(arg, ng.TensorValueOp):
                    placeholders.append(arg)
                else:
                    ops.append(arg)
            for arg in op.all_deps:
                if isinstance(arg, ng.TensorValueOp):
                    placeholders.append(arg)
                else:
                    ops.append(arg)
        return placeholders

    @staticmethod
    def get_copied_params(graph, input_params):
        placeholders = ExecutorFactory.get_all_placeholders(graph)
        copied_params = OrderedSet()
        for i in input_params:
            for p in placeholders:
                if i.name == p.tensor.name:
                    copied_params.add(p.tensor)
        return tuple(copied_params)

    def numeric_derivative(self, f, p_x, dx, *params):
        if is_flex_transformer(self.transformer):
            f_cpu = deepcopy(f)
            copied_params = self.get_copied_params(f_cpu, (p_x,) + params)
            comp = self.cpu_transformer.computation(f_cpu, *copied_params)
        else:
            comp = self.transformer.computation(f, p_x, *params)

        def helper(x, *args):
            def comp_helper(xx):
                return comp(xx, *args)

            return numeric_derivative(comp_helper, x, dx)

        return helper

    def derivative(self, f, px, *parameters):
        """
        Full derivative of f wrt placeholder px

        Arguments:
          f: TODO
          px: TODO
          parameters: TODO

        Returns:

        """
        fshape = f.axes.lengths
        xshape = px.axes.lengths

        # print "============="
        # for op in Op.ordered_ops([dfdx]):
        #     print '-----'
        #     print op, op.axes
        #     print op.args
        #     print '------'
        # print "============="

        if len(fshape) is 0:
            return self.transformer.computation(ng.deriv(f, px), px, *parameters)
        else:
            initial_adjoint = ng.placeholder(f.axes).named('adj')
            adjoint = np.zeros(fshape, dtype=f.dtype)
            dfdx = ng.deriv(f, px, error=initial_adjoint)
            comp = self.transformer.computation(dfdx, initial_adjoint, px, *parameters)

            def helper(x, *args):
                dfdxshape = list(fshape)
                dfdxshape.extend(xshape)
                npdfdx = np.empty(dfdxshape, dtype=x.dtype)

                dindex = [0 for _ in fshape]
                dindex.extend([slice(None) for _ in xshape])

                idxiter = np.nditer(
                    adjoint, flags=['multi_index'], op_flags=['readwrite'])
                for dfdxiter in idxiter:
                    dfdxiter[...] = 1

                    df = comp(adjoint, x, *args)

                    if is_flex_transformer(comp.transformer):
                        reset_flex_entries(comp)

                    # import pytest; pytest.set_trace()
                    # with open("code_sum.py", "w") as f: f.write(comp.transformer.code.code)

                    dindex[0:len(fshape)] = idxiter.multi_index
                    npdfdx[tuple(dindex)] = df
                    dfdxiter[...] = 0

                return npdfdx

            return helper


@contextmanager
def executor(results, *parameters):
    """
    Generate a single-entry transformer that computes results from parameters

    Arguments:
      results: TODO
      parameters: TODO

    Returns:
      Function of placeholders in parameters
    """
    ex = ExecutorFactory()
    ex.__enter__()
    yield ex.executor(results, *parameters)
    ex.__exit__()


def numeric_derivative(f, x, dx):
    """
    Computer df/dx at x numerically.
    Do not use for non-continuous derivatives such as min/max.  If there is a tie at the
    extremum, only one value will change and the computed derivative will be very wrong.

    Would be useful to have a batch axis some time.

    Arguments:
      f: Tensor function.
      x: Derivative position.
      dx: scalar dx change in each dimension

    Returns:
      Derivative, with f(x), x indexing, i.e. if f is 2x4 and x is 3x7, result is 2x4x3x7.
    """

    def shape(x):
        """
        Returns the shape of the tensor/scalar x
        """
        if isinstance(x, np.ndarray):
            return x.shape
        else:
            return ()

    if isinstance(x, np.ndarray) and x.dtype == int:
        raise ValueError('x shouldnt be of type int, should be a float')

    xshape = shape(x)
    # Copy because we always compute into the same place
    y = np.copy(f(x))
    fshape = shape(y)
    dshape = list(fshape)
    dshape.extend(xshape)
    d = np.zeros(shape=dshape, dtype=np.float32)
    dindex = [slice(None) for _ in fshape]
    dindex.extend((0 for _ in xshape))

    idxiter = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    for xiter in idxiter:
        old_x = np.float32(xiter)
        xiter[...] = old_x + dx
        dy = f(x) - y
        dindex[len(fshape):] = idxiter.multi_index
        d[tuple(dindex)] = (dy / dx)
        xiter[...] = old_x
    return d


def check_derivative(f, x, delta, x_value, parameters=[], parameter_values=[], **kwargs):
    """
    Check that the numeric and symbol derivatives of f with respect to x are
    the same when x has value x_value.

    Arguments:
        f: function to take the derivative of
        x: variable to take the derivative with respect to
        delta: distance to perturn x in numeric derivative
        x_value: the value of x we are going to compute the derivate of f at
        parameters: extra parameters to f
        parameter_values: value of extra parameters to f
        kwargs: passed to assert_allclose.  Useful for atol/rtol.
    """

    with ExecutorFactory() as ex:

        dfdx_numeric = ex.numeric_derivative(f, x, delta, *parameters)
        dfdx_symbolic = ex.derivative(f, x, *parameters)

        ng.testing.assert_allclose(
            dfdx_numeric(x_value, *parameter_values),
            dfdx_symbolic(x_value, *parameter_values),
            **kwargs
        )


def is_flex_transformer(transformer):
    return is_flex(transformer.transformer_name)


def is_flex_factory(transformer_factory):
    return is_flex(transformer_factory.name)


def is_flex(name):
    # Probably 'argon' also need to be added here
    flex_transformers = ['flexgpu']
    if name in flex_transformers:
        return True
    return False


def reset_flex_entries(comp):
    for flex_id in comp.executor.output_flex_ids:
        flex_entry = comp.transformer.flex_manager.flex_entries[flex_id]
        flex_entry.reset_entry()
