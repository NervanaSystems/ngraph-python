.. _transformer_usage:

.. ---------------------------------------------------------------------------
.. Copyright 2017 Intel Corporation
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

Transformers
************

Transformers are used to convert the ``Op`` graph into a backend-specific executable format. Once the graph has been defined, one or more computations are created using a transformer. Computations are handles to executable objects created by the transformer, which can be called to evaluate a subset of the entire graph. All transformers must implement a common abstract interface that allows users to easily switch between backends without altering their computation graph definition. Transformers are currently provided for the following backends:

- CPUs (via NumPy)
- NVIDIA* GPUs (via PyCUDA)

Additional transformers will be implemented for other backends in the future.

Transformer creation
====================

You should create transformers using the factory interface in ``ngraph.transformers.base``:

.. code-block:: python

    from ngraph.transformers import make_transformer
    transformer = make_transformer()

This creates a transformer using the default factory (CPU). It is possible to manually set the transformer factory to control the target backend. The transformer API provides functionality to enumerate the available transformers to assist with this:

.. code-block:: python

    import ngraph.transformers as ngt
    available_transformers = ngt.transformer_choices()
    if 'gpu' in available_transformers:
        factory = ngt.make_transformer_factory('gpu')
        ngt.set_transformer_factory(factory)

    transformer = ngt.make_transformer()

The example above first checks if the GPU transformer is available (this depends on whether CUDA and PyCuda are installed). If the GPU transformer is available, the example sets the transformer factory to generate GPU transformers. The call to ``make_transformer`` then returns a GPU transformer if one is available, and a CPU transformer otherwise.

Computations
============

Computation objects are created by the transformer and provide an interface to evaluate a subset of the graph. The format of the executable used for evaluation depends on the transformer that created the computation. For example, the CPU transformer generates Python NumPy code that is called to evaluate the computation, while the GPU transformer generates a series of CUDA kernels that can be called to evaluate the computation.

Computation creation
--------------------

Computations are created with the ``Transformer.computation`` method. When creating a computation, users must specify a list of results that should be evaluated by the computation. These results should be Intel® Nervana™ ``Op`` s. The transformer is able to traverse the graph backwards from these results to determine the entire subset of graph nodes that are required to evaluate these results, so it is not necessary for users to specify the entire subset of nodes to execute. Users must also specify a list of graph nodes to be set as inputs to the computation. Typically these are placeholder tensors. Continuing from the code example above, a simple graph and computation can be created:

.. code-block:: python

    import ngraph as ng

    a = ng.constant(4)
    b = ng.placeholder(())
    c = ng.placeholder(())
    d = ng.multiply(a, b)
    e = ng.add(d, c)

    example_comp = transformer.computation(e, b, c)

This example creates a simple graph to evaluate the function ``e = ((a * b) + c)``. The first argument is the result of the computation, and the remaining arguments are inputs to the computation. The only result that we need to specify to create the computation is ``e`` since ``d`` will be discovered when the transformer traverses the graph. In this example, ``a`` is a constant so it does not need to be passed in as an input, but ``b`` and ``c`` are placeholder tensors that must be filled as inputs.

After all computations are created, the ``Transformer.initialize`` method must be called to finalize transformation and allocate all device memory for tensors (this is called automatically if a computation is called before manually calling ``initialize``). 

.. Note::
    New computations cannot be created with a transformer after ``initialize`` has been called. For more information on this initialization process, refer to the :ref:`Transformer implementation <transformer_implementation>` documentation file.

Computation execution
---------------------

This computation object can be executed with its ``__call__`` method by specifying the input ``c``.

.. code-block:: python

    result_e = example_comp(2, 7)

The return value of this call is the resulting value of ``e``, which should be ((4 * 2) + 7) = 15.

Computations with multiple results
----------------------------------

In real world cases, we often want computations that return multiple results. For example, a single training iteration might compute both the cost value and the weight updates. Multiple results can be passed to computation creation in a list. After execution, the computation returns a tuple of the results:

.. code-block:: python

    example_comp2 = transformer.computation([d, e], b, c)
    result_d, result_e = example_comp2(2, 7)

In addition to returning the final result as above, this example also sets ``result_d`` to the result of the ``d`` operation, which should be 8.

Transformed graph state
-----------------------

Once the transformer has been initialized and the computation objects have been finalized, all tensors (constants, variables, placeholders) will be allocated in device memory. These tensors are only allocated and initialized once at transformation time, so the transformed graph has a state that is persistent between computation evaluations. This is most important for variable tensors, since constants are never modified after creation and placeholders are usually filled by the caller each time a computation is run. The value of variable tensors will remain unchanged between the completion of one computation and the subsequent evaluation of another.

Computations created by the same transformer will share state for any op graph nodes that are needed by both computations. If a variable tensor is assigned in one computation, the updated value is seen by a subsequent call to a different computation which references that variable tensor. An example of this is a script which defines both a train and test computation. We want to evaluate the test computation to check convergence periodically using the parameters that are being trained in the train computation.

Executor utility
================

For convenience, an executor utility is provided in ``ngraph.util.utils``. This executor utility reduces the process of creating a transformer and a computation to a single function call. 

.. Note::
   Calling this function creates a new transformer each time, so it should not be used for cases where multiple computations with a shared state are needed.

.. code-block:: python

    from ngraph.util.utils import executor
    example_comp = executor(e, b, c)
    result_e = example_comp(2, 7)

Graph execution
===============

A *computation* is a subset of ops whose values are desired and which correspond to a callable procedure on a backend.
Users define one or more computations by specifying sets of ops to be computed. In addition, the transformer
defines four additional procedures:

*allocate*
    Allocate required storage required for all computations. This includes all allocations for all ops
    marked as `in`.

*initialize*
    Run all initializations. These are all the `initializers` for the ops needed for the computations.  These
    are analogous to C++ static initializers.

*save*
    Save all persistent state. These are states with the `persistent` property set.

*restore*
    Restore saved state.