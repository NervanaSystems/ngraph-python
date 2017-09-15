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

Transformers
************

Transformers are used to convert the ``Op`` graph into a backend specific executable format. Once the graph has been defined, one or more computations are created using a transformer. Computations are handles to executable objects created by the transformer, which can be called to evaluate a subset of the entire graph. All transformers must implement a common abstract interface allowing users to easily switch between backends without altering their computation graph definition. Transformers are currently provided for the following backends:

- CPUs (via NumPy)
- NVIDIA GPUs (via PyCUDA)

Additional transformers will be implemented for other backends in the future.

Transformer Creation
====================

Transformers should be created using the factory interface in ngraph.transformers.base

.. code-block:: python

    from ngraph.transformers import make_transformer
    transformer = make_transformer()

This will create a transformer using the default factory (cpu). It is possible to set the transformer factory manually to control the target backend. The transformer API provides functionality to enumerate the available transformers to assist with this

.. code-block:: python

    import ngraph.transformers as ngt
    available_transformers = ngt.transformer_choices()
    if 'gpu' in available_transformers:
        factory = ngt.make_transformer_factory('gpu')
        ngt.set_transformer_factory(factory)

    transformer = ngt.make_transformer()

The above example first checks if the GPU transformer is available (this will depend on whether CUDA and PyCuda are installed). If the GPU transformer is available, the example sets the transformer factory to generate GPU transformers. The call to ``make_transformer`` will then return a GPU transformer if available, and a CPU transformer otherwise.

Computations
============

Computation objects are created by the transformer and provide an interface to evaluate a subset of the graph. The format of the executable used for evaluation depends on the transformer that created the computation. For example the CPU transformer generates python NumPy code which is called to evaluate the computation, while the GPU transformer generates a series of CUDA kernels which can be called to evaluate the computation.

Computation Creation
--------------------

Computations are created with the ``Transformer.computation`` method. When creating a computation, the user must specify a list of results which should be evaluated by the computation. These results should be ngraph ``Op`` s. The transformer is able to traverse the graph backwards from these results to determine the entire subset of graph nodes required to evaluate these results, so it is not necessary for the user to specify the entire subset of nodes to execute. The user must also specify a list of graph nodes to be set as inputs to the computation. Typically these are placeholder tensors. Continuing from the above code example, a simple graph and computation can be created:

.. code-block:: python

    import ngraph as ng

    a = ng.constant(4)
    b = ng.placeholder(())
    c = ng.placeholder(())
    d = ng.multiply(a, b)
    e = ng.add(d, c)

    example_comp = transformer.computation(e, b, c)

This example creates a simple graph to evaluate the function ``e = ((a * b) + c)``. The first argument is the result of the computation and the remaining arguments are inputs to the computation. The only result that we need to specify to create the computation is ``e`` since ``d`` will be discovered when the transformer traverses the graph. In this example, ``a`` is a constant so it does not need to be passed in as an input, but ``b`` and ``c`` are placeholder tensors which must be filled as inputs.

Computation Execution
---------------------

This computation object can be executed with its ``__call__`` method by specifying the inputs ``b`` and ``c``.

.. code-block:: python

    result_e = example_comp(2, 7)

The return value of this call will be the resulting value of ``e``, which should be ((4 * 2) + 7) = 15.

Computations with Multiple Results
----------------------------------

In real world cases, we often want computations that return multiple results. For example a single training iteration may compute both the cost value and the weight updates. Multiple results can be passed to computation creation in a list. After execution, the computation will return a tuple of the results:

.. code-block:: python

    example_comp2 = transformer.computation([d, e], b, c)
    result_d, result_e = example_comp2(2, 7)

In addition to returning the final result as above, this example will also set ``result_d`` to the result of the ``d`` operation, which should be 8.

Transformed Graph State
-----------------------

Once the transformer has been initialized and computation objects have been finalized, all tensors (constants, variables, placeholders) will be allocated in device memory. These tensors are only allocated and initialized once at transformation time, so the transformed graph has state that is persistent between computation evaluations. This is most important for variable tensors, since constants are never modified after creation and placeholders are usually filled by the caller each time a computation is run. The value of variable tensors will remain unchanged between the finish of one computation and the subsequent evaluation of another.

Computations created by the same transformer will share state for any op graph nodes which are needed by both computations. If a variable tensor is assigned in one computation, the updated value will be seen by a subsequent call to a different computation which references that variable tensor. An example of this is a script that defines both a train and test computation. We want to evaluate the test computation to check convergence periodically using the parameters being trained in the train computation.

Graph execution
===============

A *computation* is a subset of ops whose values are desired and corresponds to a callable procedure on a backend.
Users define one or more computations by specifying sets of ops to be computed.  In addition, the transformer
will define four additional procedures:

`allocate`
    Allocate required storage required for all computations.  This includes all allocations for all ops
    marked as `in`.

`initialize`
    Run all initializations.  These are all the `initializers` for the ops needed for the computations.  These
    are analogous to C++ static initializers.

`save`
    Save all persistent state.  These are states with the `persistent` property set.

`restore`
    Restore saved state.
