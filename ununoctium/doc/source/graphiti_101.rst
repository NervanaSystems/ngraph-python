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

.. include:: <isonum.txt>

Graphiti 101
************

Nervana's graph architecture has three layers, the model description, the operational graph,
and the MOP (Machine-learning OPerations).  Models are described using higher-level APIs such
as neon or TensorFlow\ |trade|.  Front-ends specific to the API use the operational graph API
to convert the model description into an operational graph.  Back-ends correspond to compute
platforms, such as GPUs or CPUs.  A combination of generic and
back-end specific transformers optimize and the operational graph and convert it to a form
that can be run on the back-end.  The operational graph API is shared by all front-ends and
back-ends.

The operational graph API can be used directly, both as an alternative to a front-end and as
an extension mechanism for front-ends.