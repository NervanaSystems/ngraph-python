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

The term *transformer* refers to all operations related to running computations, from compilation to execution, that are defined by a graph on a backend. Since any subset of the graph represents a potential computation, the graph should be thought of as a template for computations related to some particular model. For example, the core of a model is its inference graph. An optimizer extends the inference graph by adding the derivative computations and variable updates. The extended graph includes the computations used for both inference and for training.

The transformer method ``computation`` is used to specify a subgraph to be computed. This method needs one or more result graph nodes that need to be computed; their values will be returned when the computation is executed. The computation also needs a parameter list of nodes that will receive the arguments when the computation is called. This parameter list must include all the ``placeholder`` nodes that contribute to the computed nodes. You can include include additional nodes, such as variables, by setting the ``is_input`` attribute to ``True``. The ``computation`` method returns a function that expects tensors for each parameter and returns tensors for each value-producing result node.

Transformers are currently provided for the following backends:

- CPUs (via NumPy and MKL-DNN)
- NVIDIA* GPUs (via PyCUDA)

Transformer creation
====================

You should create transformers using the factory interface in ``ngraph.transformers.base``, as shown:

.. code-block:: python

    from ngraph.transformers import make_transformer
    transformer = make_transformer()

This creates a transformer using the default transformer factory (CPU). You can manually set the transformer factory to control the target backend. The transformer API provides functionality to enumerate the available transformers to assist with this, as shown below:

.. code-block:: python

    import ngraph.transformers as ngt
    available_transformers = ngt.transformer_choices()
    if 'gpu' in available_transformers:
        factory = ngt.make_transformer_factory('gpu')
        ngt.set_transformer_factory(factory)

    transformer = ngt.make_transformer()

The example above first checks if the GPU transformer is available (this depends on whether CUDA and PyCUDA are installed). If the GPU transformer is available, this example sets the transformer factory to generate GPU transformers. The call to ``make_transformer`` then returns a GPU transformer if one is available. Otherwise, the ``make_transformer`` call returns a CPU transformer.

Computations
============

Computation objects are created by the transformer and provide an interface to evaluate a subset of the graph. The format of the executable used for evaluation depends on which transformer created the computation object. For example, the CPU transformer generates Python NumPy code that is called to evaluate the computation, while the GPU transformer generates a series of CUDA kernels that can be called to evaluate the computation.

Computation creation
--------------------

Computations are created with the ``Transformer.computation`` method. When creating a computation, users must specify a list of results that should be evaluated by the computation. These results should be Intel® Nervana™ ``Op`` s. The transformer can traverse the graph backwards from these results to determine the entire subset of graph nodes that are required to evaluate these results. This means that is not necessary for users to specify the entire subset of nodes to execute. However, users must set a list of graph nodes as inputs to the computation. Typically these are placeholder tensors. Continuing from the code example above, let's create a simple graph and computation:

.. code-block:: python

    import ngraph as ng

    a = ng.constant(4)
    b = ng.placeholder(())
    c = ng.placeholder(())
    d = ng.multiply(a, b)
    e = ng.add(d, c)

    example_comp = transformer.computation(e, b, c)

This example creates a simple graph to evaluate the function ``e = ((a * b) + c)``. The first argument is the result of the computation, and the remaining arguments are inputs to the computation. To create the computation, the only result that we need to specify is ``e``, since ``d`` is discovered when the transformer traverses the graph. In this example, ``a`` is a constant so it does not need to be passed in as an input, but ``b`` and ``c`` are placeholder tensors that must be specified as inputs.

After any computations is created, the ``Transformer.initialize`` method must be called to finalize transformation and allocate all device memory for tensors (the ``Transformer.initialize`` method is called automatically if a computation is called before you manually call ``initialize``). 

Computation Execution
---------------------

Our example computation object can be executed with its ``__call__`` method by specifying the inputs ``b`` and ``c``.

.. code-block:: python

    result_e = example_comp(2, 7)

The return value of this call is the resulting value of ``e``, which should be ((4 * 2) + 7) = 15.

Computations with multiple results
----------------------------------

In real world use cases, we often want to create computations that return multiple results. For example, a single training iteration might compute both the cost value and the weight updates. Multiple results can be passed to computation creation in a list. After execution, the computation returns a tuple of the results:

.. code-block:: python

    example_comp2 = transformer.computation([d, e], b, c)
    result_d, result_e = example_comp2(2, 7)

In addition to returning the final result as seen above, this example also sets ``result_d`` to the result of the ``d`` operation, which should be 8.

Transformer/Backend state
-------------------------

A computation is compiled and installed on the backend device the first time the computation is called. Any new persistent tensors (such as variables) will be initialized at this time. Persistent tensors that were also used in previously defined computations will retain their states unless they have been listed among the computation's arguments. If some persistent tensors are listed among the computation's arguments, their values will be set when the computation is invoked. For example, variables updated by a training computation will retain their values for an inference computation. You can manually save variables by defining a computation that returns their values, and can store variables by using them as arguments for a computation.

