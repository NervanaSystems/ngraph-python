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

Overview
********

|Geon| is a Python library for implementing programs that convert descriptions of neural networks into programs
that run efficiently on a variety of platforms. The library has three layers: frontends, the operational graph (or op-graph), and transformers. The work flow is shown below:

.. image :: assets/ngraph_workflow.png

Frontends in |geon| such as neon or |TF| provide methods to users for defining models using high level building blocks such as layers and optimizers in the case of neon or lower level computations in the case of |TF|. These frontends themselves use the Nervana Graph API to build a directed acyclic graph (DAG) composed of nodes called ``Op``\s linked by edges representing their data dependencies. Nervana Graphs are composed of hardware-independant ``Op``\s.

After providing one or more subsets of these ``Op``\s as computations, a transformer is then used to optimize and compile the computations for execution. The transfomer also produces functions for allocation, initialization, and saving/restoring models. Transformers can be device-specific, from single or multiple CPUs or GPUs to mobile devices or large distributed servers.

The |geon| APIs are factored into an API for frontends and an API for transformers.  The frontend API can also be directly used to construct a model as an operational graph, as is done by a number of unit tests.
