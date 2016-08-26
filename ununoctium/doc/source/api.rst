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

This API documentation covers each module within Graphiti.

geon.backends
=============
.. py:module: geon.backends

.. autosummary::
   :toctree: generated/
   :nosignatures:

   geon.backends.dataloaderbackend
   geon.backends.graph.artransform
   geon.backends.graph.environment
   geon.backends.graph.mpihandle

geon.frontends
==============
.. py:module: geon.frontends

.. autosummary::
   :toctree: generated/
   :nosignatures:

   geon.frontends.base.axis
   geon.frontends.base.graph
   geon.frontends.declarative_graph.declarative_graph

geon.frontends.neon
===================
.. py:module: geon.frontends.neon

.. autosummary::
   :toctree: generated/
   :nosignatures:

   geon.frontends.neon.activation
   geon.frontends.neon.callbacks
   geon.frontends.neon.container
   geon.frontends.neon.cost
   geon.frontends.neon.layer
   geon.frontends.neon.model
   geon.frontends.neon.optimizer

geon.op_graph
=============
.. py:module: geon.op_graph

.. autosummary::
   :toctree: generated/
   :nosignatures:

   geon.op_graph.arrayaxes
   geon.op_graph.convolution
   geon.op_graph.names
   geon.op_graph.nodes
   geon.op_graph.op_graph

geon.transformers
=================
.. py:module: geon.transformers

.. autosummary::
   :toctree: generated/
   :nosignatures:

   geon.transformers.base
   geon.transformers.nptransform

geon.util
=========
.. py:module: geon.util

.. autosummary::
   :toctree: generated/
   :nosignatures:

   geon.util.generics
   geon.util.graph
   geon.util.pygen
   geon.util.threadstate
   geon.util.utils

geon.analysis
=============
.. py:module: geon.analysis

.. autosummary::
   :toctree: generated/
   :nosignatures:

   geon.analysis.dataflow
   geon.analysis.fusion
   geon.analysis.memory

