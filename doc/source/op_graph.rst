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

Operational Graph
*****************

The operational graph, or op-graph, is the intermediate representation for |geon| computations. A transformer defines primitive operations that it can easily convert into code that runs on a combination of the CPU and device. A front end converts a model description into operations that can later be transformed into primitive operations by replacing subgraphs with more primitive but semantically equivalent subgraphs. Frontends do this by calling functions which return graph ``Ops``, nodes in the graph that correspond to the computation of the node and all nodes it depends on.

The op-graph's purpose is to simplify the implementation of frontends and transformers, so it provides a core of semi-primitive operations, higher-level operations, and transformations from the higher-level operations to the semi-primitive operations. These are called the standard operations and transforms. A frontend developer may define additional operations and transformations to the standard operations, and a transformer developer may provide additional primitive operations and transforms from standard operations to the primitive operations. This also provides a mechanism for exposing transformer-specific operations to a frontend, similar to the way some compilers permit inline assembly.

Standard Operations
===================

The standard operations are partitioned into groups of operations:

Semi-primitive
    Semi-primitive operations are simple to implement. In some cases, a semi-primitive operation can be transformed into more primitive semi-primitive operations.  A transformer specifies a base subset of the semi-primitive opertions; no semi-primitive operations in the base set will be transformed by standard transformations.  However, a transformer may add transforms that transform operations in the base set. For example, it may add transforms that convert base set operations into transformer-specific operations, or into other semi-primitive operations in a manner better suited to its backend.  Semi-primitive operations work with low-dimensional row-major tensors.

    Not all operations take place on the device. For example, storage allocation and view creation occurs on the CPU during transformation. Thus, some semi-primitive operations are for indicating where CPU operations are needed. There are also semi-primitive operations for transferring data to/from the device.

NumPy
    NumPy operations correspond to a subset of NumPy. They are intended for frontends in which all tensor shaping is performed by the model developer.

Simple Axes
    Some backends are more efficient with the sample axis first, some with it last. When a frontend can determine the sample axis, it may use the simple axes operations, so that a transformer can enable the |ngraph| transforms that favor the desired sample axis position.  Similarly, for RNNs the optimal positioning of the R axis might be dependent on the backend, so the simple axes API support indication of the R axis.

Axes
    The Axes operations use abstract indexing. A tensor has a set of abstract indices, and tensor operations match tensor indices.  This makes it considerably simpler to work with tensors in a dimension-free manner for higher-level frontends that can capture the semantics of the tensor indices.

Graph Implementation
====================

Functions that return op-graph nodes, ``Ops``, are always used to build the graph from other ``Ops``. Frontends use the ``Ops`` as handles to the graph.  In Python, heavy use of *magic methods* such as ``__add__`` is made, to simplify the construction of the graph; for example, if ``x`` and ``y`` are ``Ops`` then ``x + y`` will be an ``Op`` for adding ``x`` and ``y``. Just which form of ``add`` is chosen depends on which family of standard operations created ``x`` and ``y``.

Many ``Ops`` correspond to operations that compute tensors.  These ``Ops`` have a *tensor description* that describes the tensor that the ``Op`` returns.  A tensor description is essentially a type for the tensor. Tensor descriptions for semi-primitive ``Ops`` are different from tensor descriptions for NumPy ``Ops``, etc., and the ``Op`` classes are also different, so that the magic methods create the correct ``Ops``.  In general, coercion ``Ops`` are needed when going from one group of operations to another.  These can be inserted automatically, or an exception can be raised when incompatible ``Ops`` are used as arguments.

Every ``Op`` has an ``args`` attribute which is a sequence of ``Ops`` that must be executed before the ``Op``. An ``Op`` also has an attribute ``other_deps``, which is a set of ``Ops`` besides ``args`` that should be executed before the ``Op``. In addition, ``Ops`` have a ``user_deps`` attribute, another set of ``Ops``. All the ``user_deps`` of the ``args`` of an ``Op`` are added to the ``other_deps`` of the ``Op`` when the ``Op`` is created.  When a tensor variable is modified, the tensor variable's ``user_deps`` is set to the modification ``Op`` to ensure that the modification happens before the value is used.

The ``Op`` nodes are used as handles for obtaining tensor values op operations after transformation.  In normal execution, only the nodes specifically requested will have valid values, but if stepped debugging is supported by a transformer then the nodes can be used to probe values during a computation.

Since the transformation process replaces subgraphs with more primitive subgraphs, it is likely that a frontend holds nodes that are no longer part of the graph after the transformation process, but the frontend still needs to use these nodes as handles to the values. The transformer does not have access to the frontend to replace the nodes it is holding with their new counterparts.  We adopt a simple technique used in some Lisp implementations: When replacing a subgraph with a new subgraph, the ``forward`` attribute of each ``Op`` in the replaced subgraph is set to an ``Op`` with the same value in the replacement.  When the ``Op``'s value is needed by the frontend, it follows ``forward`` attributes until it reaches an ``Op`` where ``forward`` is ``None``, and uses that node.

Sometimes a node can disappear.  For example, a ``log(exp(x))`` could be replaced by ``x``, but a frontend could be holding the ``log`` or ``exp`` ``Op``.  In this case, there are a few choices for the ``forward.`` If the ``Op`` is one that has been specified as needing to be available, the transformation should ensure that the value is still computed.  Otherwise, the ``forward`` attribute can be set to ``Invalid`` or it can be set to CPU operations that compute it from ``x``.

When one subgraph replaces another, the ``args`` and ``other_deps`` for all extenral ``Ops`` that reference the subgraph are updated (``user_deps`` is only used during graph construction), so only the frontend value access functions need to make use of the ``forward`` attribute.  The ``snap`` function will follow forwarding; a frontend may use this function to pre-forward ``Ops``.  In the case where the ``Op`` forwards to an ``invalid``, the original ``Op`` is returned, so that an exception is only raised if an attempt is made to access the value.

Schema
======

In some cases, it is convenient for a op-graph function to associate additional information with an ``Op``. For example, the ``softmax`` function returns a division ``Op`` but when the value is used in cross-entropy, the computation will be numerically unstable if performed directly.  The ``softmax`` function can indicate that the ``division`` is part of a ``softmax`` computation and indicate the sub-graphs that are useful in cross-entropy and derivatives by adding a schema to the division ``Op``.

