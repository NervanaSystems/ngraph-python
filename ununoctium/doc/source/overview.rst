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
that run efficiently on a variety of platforms.  If you are primarily developing models, |geon| won't directly help
you, but it might help you extend a front end such as neon to make it easier to implement new kinds of models.
On the other hand, if you are developing new ways to specify models, developing tools for analyzing and debugging
models, or have a new kind of compute platform for training or inference of your models, |geon| is intended to
help you.

As an example, consider two APIs for developing models, neon and |TF|.  Each corresponds to a |geon| front end.
Models developed with the front ends might be trained on a variety of platforms; single or mulitiple CPUs or
GPUs, and inference could be performed on anything from mobile devices to large distributed servers.  Each of these
corresponds to a |geon| back end developed independently of the |geon| front ends.

In between the front ends and the back ends is the |geon| operational graph, or opgraph.  The opgraph is a
flow graph of operations, with nodes, called ``Ops``, corresponding to tensor values of primitive tensor operations,
and arcs corresponding to inputs to the operations.  Front ends translate model descriptions into an opgraph.
Next the front end chooses a *transformer* for the desired back end and specifies one or more subsets of ``Ops``
as *computations*.  When the desired computations have been specified, the transformer compiles the computations
into functions.  In addition, the transformer produces functions for allocation, initialization, saving a model,
and restoring a saved model.

As a concrete example, for training a front end might want two computations, one for processing a batch for update,
and one for evaluating a batch for a running report of training progress.  To perform training, the allocation
function would be called, followed by the initialization function, and then multiple batch update functions
interspersed with batch evaluation functions.  Longer computations would intersperse calls to the state saving
function for checkpointing purposes.  If traning were interrupted, allocation would be followed by a restore
rather than initialization.

The |geon| APIs are factored into an API for developing front ends and an API for developing transformers for back
ends.  Since front ends use the front end API to construct an operational graph, define computations, and compile
models, the front end API can also be directly used to construct a model as an operational graph, as is done by
a number of unit tests.