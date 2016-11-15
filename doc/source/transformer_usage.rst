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
