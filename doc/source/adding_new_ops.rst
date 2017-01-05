.. ---------------------------------------------------------------------------
.. Copyright 2016 Nervana Systems Inc.
.. Licensed under the Apache License, Version 2.0 (the "License");
.. you may not use this file except in compliance with the License.
.. You may obtain a copy of the License at
..
..      http://www.apache.org/licenses/LICENSE-2.0
..
.. Unless required by applicable law or agreed to in writing, software
.. distributed under the License is distributed on an "AS IS" BASIS,
.. WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
.. See the License for the specific language governing permissions and
.. limitations under the License.
.. ---------------------------------------------------------------------------

Adding New Ops
**************

Overview
--------
To add a new op in ngraph, you'll need to :

- Register the new op in ``op_graph``
- Add adjoint function for computing gradients of the op (optional)
- Register op in transformer passes
- Add implementation in transformers (such as numpy and gpu)
- Add corresponding tests

Example
-------
In the following example, we'll walk though the steps for adding the ``ng.prod``
op.

1. First, we need to register the new op in ``op_graph``. In general, an op is
   a sub-class of ``ngraph.op_graph.op_graph.Op``. To add a new op, we could
   inherit from the class ``Op`` or one of its descendant classes and implement
   the functionalities correspondingly. During the op registration phase, we
   set up axes and other necessary attributes / functions for op implementation
   in transformers. Since ops of different types have distinct functionalities,
   we could usually find references on ops that have similar behavior or belong
   to the same category.

   In this case, since ``Prod`` is a reduction op, we could use the
   ``create_reduction_op`` helper function. So in
   ``ngraph/op_graph/op_graph.py``, we add ::

        Prod, ProdTwoDim, ProdOneDim, prod = create_reduction_op(
              'Prod', 'ProdTwoDim', 'ProdOneDim', 'prod', prod_adjoints
        )

2. Next, we could (optionally) add adjoint function for computing gradients of
   the op in ``Op.generate_adjoints()`` function. There are two scenarios: if
   the gradients of the op could be represented by other ops available in
   ngraph, we could use those ops to implement the gradients; if that is not
   possible or if we want to optimize the performance of the gradient
   computation, we could add new ops specifically for computing gradient.

   In this example, we could represent the gradeints of the ``Prod`` by other
   ngraph ops. In ``ngraph/op_graph/op_graph.py``, we add ::

        def prod_adjoints(self, adjoints, delta, x):
            # axes
            axes = x.axes
            reduction_axes = self.reduction_axes

            # x_equal_zero
            x_equal_zero = equal(x, 0)

            # count 0's occurrence by reduction axes
            x_zero_count = sum(x_equal_zero, reduction_axes=reduction_axes)

            # create mask for zero count 0 and 1
            mask_zero = broadcast(equal(x_zero_count, 0), axes=axes)
            mask_one = broadcast(equal(x_zero_count, 1), axes=axes)

            # replace all 0 to 1
            x_replaced = equal(x, 0.) * 1. + (1. - equal(x, 0.)) * x

            # do product of x_replace and gradient
            x_replaced_prod = prod(x_replaced, reduction_axes=reduction_axes)
            x_replaced_grad = x_replaced_prod / x_replaced

            # multiply mask with mask for the two cases
            x_grad = mask_zero * x_replaced_grad + mask_one * x_equal_zero * x_replaced_grad

            x.generate_add_delta(
                adjoints,
                broadcast(delta, x.axes) * x_grad
            )

3. The next step is to register op in transformer passes. Transformer passes
   are used to simplify graph, optimize ops for execution and meet device
   specific constraints. The two default passes we have currently are
   ``SimplePrune`` and ``RequiredTensorShaping``.

   For ``Prod``, one of the optimization we can do is that, if the tensor are
   filled with the identical value, we could replace it by the ``Power`` op.
   Therefore, in ``ngraph/transformers/passes/passes.py``, we add ::

        class RequiredTensorShaping(PeepholeGraphPass):
            ...

            @visit.on_type(Prod)
            def visit(self, op):
                x = op.args[0]
                if x.is_scalar:
                    # Prod of a scalar is just the scalar raised to the power of the
                    # axes size rebroadcast
                    val = broadcast(power(cast_axes(x, ()), op.reduction_axes.size), op.axes)
                    self.replace_op(op, val)
                    return
                # call-next-method
                if op.must_reduce:
                    self.replace_op(op, op.reduce_to_twod())

        class SimplePrune(PeepholeGraphPass):
            ...

            @visit.on_type(Prod)
            def visit(self, op):
                """
                TODO.
                Arguments:
                  op: TODO
                Returns:
                  TODO
                """
                x, = op.args
                if x.is_scalar and x.is_constant:
                    val = power(x.const, op.reduction_axes.size)
                    self.replace_op(op, constant(val))

4. Next, we need to add implementations of the op in transformers. Notes that
   in the previous steps, we still haven't specified how the op shall be executed
   (forward computation). In current ngraph, the ops are implemented in
   ``NumpyTransformer`` and ``GPUTransformer`` are done by code generation for
   optimized performance.

   In ``ngraph/transformers/nptransform.py``, add the following for numpy
   code generation ::

        class NumPyCodeGenerator(PyGen):
            ...

            @generate_op.on_type(Prod)
            def generate_op(self, op, out, x):
                self.append("np.prod({}, axis=0, out={})", x, out)

   In ``ngraph/transformers/gputransform.py``, add ::

        class ElementWiseKernel(GPUKernel):
            ...

            @add_op.on_type(Prod)
            def add_op(self, op, out, x):
                self._buffer_op("prod", x=x, axis=0, out=out)

   Finally in ``/ngraph/transformers/gpu/float_ew2.py`` add the following for
   the reduction op generation template ::

        _redop_templates = {
            "prod": r"%(out)s = %(out)s * %(x)s;",
            ...
        }

        _redop32_templates = {
            "prod": r"%(out)s = %(out)s * __shfl_xor(%(out)s, i);",
            ...
        }

        _redop_inits = {
            "prod": "1.0f",
            ...
        }

5. The last step is to add the corresponding tests to test the correctness
   of the forward and backward computation. For ``ng.prod``, please refer to
   ``test_prod_constant()`` and ``test_prod_deriv`` test function under
   ``tests/test_execution.py``.
