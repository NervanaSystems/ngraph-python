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

Transformer Implementation
**************************

This document gives an overview of how the base transformer and computation is implemented, using the NumPy and GPU transformers as examples.

Transformer Creation
====================

The base transformer constructor initializes a set of all computations and results associated with the transformer. As computation objects are created, these sets are populated. Additionally the transformer constructor can build a list of passes to run on the op graph when initialization and transformation is executed.

Specific transformer implementations may use the constructor to initialize code generators (as in the NumPy transformer) or initialize the target device and determine device capabilities (as for the GPU transformer).

Computation Creation
====================

To create a computation, a user calls the transformer's ``computation`` method. This is a relatively lightweight operation that creates a new ``Computation`` object and stores it in the set of all computations. The ``Computation`` constructor updates the transformer's set of results and builds a set of all ops that are dependencies of the results by traversing the graph backwards from the result nodes.

Transformer Initialization
==========================

The ``Transformer::initialize`` method of the transformer is responsible for running passes to augment the graph, generating code or executables to evaluate ops, allocating buffers and tensors, and initializing tensors. This method can be called manually by the user, but will be automatically called upon the first evaluation of a computation if the user has not manually called it.

Passes and op transformation are called from ``Transformer::_transform_computations``. Device buffer and tensor allocation is called from ``Transformer::allocate_storage`` which must be implemented by each transformer. Constant tensor initialization is called from ``Transformer::allocate`` and other initialization is performed in a special computation called by ``Transformer::initialize``.

Transformer Passes
------------------

Transformer passes are run in ``Transformer::_transform_computations`` here:

.. code-block:: python

    def _transform_computations(self):
        """
        Transform computation graphs to a form that can be run.
        """

        # with Op.saved_user_deps():
        # Run passes on the computation graphs
        self.run_registered_graph_passes(self.all_results)

Transformer passes are used to replace ops in the graph, remove ops from the graph, or splice ops into the graph. These passes can be used to simplify the graph (see ``SimplePrune`` for an example). Passes can also be used to alter the graph to meet device specific constraints or to optimize ops for exection on specific devices. Currently the only pass that falls into this category is the ``RequiredTensorShaping`` pass which reduces the dimensionality of tensors to 2d for reduction elementwise operations and 1d for all other elementwise operations. This pass simplifies the requirements placed on code generation to handle multi-dimensional tensors. In the future, we will likely make this device specific (for example, the GPU kernel generator can handle up to 3 dimensions efficiently).

All passes inherit from the ``GraphPass`` class which requires that a child class implement ``do_pass``. Currently all implemented passes are instances of ``PeepholeGraphPass``. A peephole graph pass is a specific type of pass that traverses the graph one node at a time, calling ``PeepholeGraphPass::visit`` on each node and builds a mapping of ops to be replaced. Implementors can define ``visit`` methods for relevant op types and call ``PeepholeGraphPass::replace_op`` to replace the visited op with another op. An example from the ``SimplePrune`` pass is shown below:

.. code-block:: python

    @visit.on_type(Add)
    def visit(self, op):
        """
        TODO.

        Arguments:
          op: TODO

        Returns:
          TODO
        """
        x, y = op.args
        rep = None
        if x.is_scalar and x.is_constant:
            if x.const == 0:
                rep = y
        elif y.is_scalar and y.is_constant:
            if y.const == 0:
                rep = x
        if rep is not None:
            self.replace_op(op, rep)

The ``@visit.on_type(Add)`` line indicates that this method will be called when an ``Add`` op is encountered during graph traversal. This implementation of ``visit`` is checking if either of the arguments to ``Add`` is 0. Since adding 0 to a value is essentially a no-op, an op meeting this condition can be replaced with its non-zero argument.

Passes present a major opportunity for performance optimization that we plan to exploit in the future. This will likely include device specific fusion of operations that will allow generation of kernels to execute multiple ops at once and buffer sharing that will allow non-overlapping operations to share device memory. These passes would improve execution time and memory usage respectively.

Intialization Computation
-------------------------

A special initialization computation is created in ``Transformer::_transform_computations`` which is responsible for executing any initializers attached to graph ops. Initializers are discovered and enumerated in ``Transformer::ordered_initializers`` by tranversing the graph and checking for the ``Op::initializers`` member. This set of initializers is used to construct the initialization computation

.. code-block:: python

        # Collect up all ops from the graph and obtain the init graph
        all_ops = OrderedSet(Op.ordered_ops(self.all_results))
        init_op = doall(self.ordered_initializers(all_ops))

        ...

        # create computation which initializes values (called once per session)
        init_op.update_forwards()
        self.init_computation = self.computation(init_op, name="init")

This computation is then transformed in the same manner as other computations and run later in ``Transformer::initialize``.

Tensor Description Initialization
---------------------------------

Tensor descriptions for all ops are initialized in ``Transformer::initialize_tensor_descriptions``. This calls into the transformer to create ``DeviceBufferStorage`` and ``DeviceTensor`` instances for each op. Each transformer must define implementations of ``DeviceBufferStorage`` and ``DeviceTensor``.

The ``DeviceBufferStorage`` class represents a memory allocation on the transformer's device (for example this will be allocated with PyCUDA for the GPU transformer). This buffer can be used a storage by one or more tensors. When a ``DeviceBufferStorage`` object is created, the buffer is not allocated yet, but the object is added to the ``Transformer::device_buffers`` member for later allocation.

The ``DeviceTensor`` class represents a tensor view on top of a device memory allocation including a base address offset, shape, strides, and data type. A ``DeviceTensor`` object is created for every ``TensorDescription`` in the graph during ``Transformer::initialize_tensor_descriptions``. When a ``DeviceTensor`` object is created, the individual transformer can handle it in multiple ways. The NumPy and GPU transformers both tag ``DeviceTensor`` objects to their underlying ``DeviceBufferStorage`` objects so that they can be allocated at the same time as the device allocation. Each transformer's ``DeviceTensor`` implementation must support some simple operations including copying to and from NumPy arrays. This is used to set argument values in the graph and get result values from the graph.

After all tensor descriptions are initialized and have created their device buffers and tensors, their allocation is transformed:

.. code-block:: python

        self.start_transform_allocate()
        for device_buffer in self.device_buffers:
            device_buffer.transform_allocate()
        self.finish_transform_allocate()

What this means is that the actual allocation of buffers and tensors is transformed into an executable format similar to computations so that it can be run later. This transformed allocation operation is eventually executed by the ``Transformer::allocate_storage`` method.

Computation Transformation
--------------------------

Computation objects are finally transformed into an executable format after allocations are transformed in ``Transformer::_transform_computations``

.. code-block:: python

        for computation in self.computations:
            computation.transform()

The ``Computation::transform`` method first gets the set of all ops needed to evaluation the computation. Since graph passes may have replaced ops by updating their forward pointers, this method will get the fully forwarded set of ops. Then the ops are ordered in such a way that all execution dependencies are met using ``Digraph::can_reach``.

Each transformer implements a ``Transformer::transform_ordered_ops`` which accepts a list of ordered ops and transforms them into an executable format. The NumPy transformer implements this by generating a python function containing one or more NumPy calls for each op. Individual ops are handled in the NumPy transformer with the corresponding ``NumPyCodeGenerator::generate_op`` implementation. The GPU transformer implements this by generating a ``GPUKernelGroup`` containing a set of ``GPUKernel`` objects which can be executed to evaluate each op. Individual ops are handled in the GPU transformer with the corresponding ``GPUKernelGroup::add_kernel`` implementation or ``ElementWiseKernel::add_op`` implementation. The ElementWiseKernel generates CUDA C code to evaluate most op types. Other more complex ops have hand-written GPU kernels such as convolution and GEMM. These are handled in different ``GPUKernel`` implementations.

When transformation of computations has finished, the transformer implementation must set the ``Computation::executor`` member to either a function or callable object which will serve as the entry point for computation evaluation.

Computation Execution
=====================

Computations are executed by calling the ``Computation::executor`` member. For the NumPy transformer this is a function pointer to the corresponding function in the generated python NumPy code. For the GPU transformer this is the corresponding ``GPUKernelGroup`` object which implements the ``__call__`` method.
