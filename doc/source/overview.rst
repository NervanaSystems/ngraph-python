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
========

:Release: |version|
:Date: |today|

.. Note::
   Nervana Graph is currently a preview release and the API's and implementation are subject to change. We encourage you to contribute to the discussion and help shape the future Nervana Graph.

The Nervana Graph (ngraph) is a Python library for implementing programs that convert descriptions of neural networks into programs that run efficiently on a variety of platforms. In designing ngraph we kept three guiding motivations in mind:

1. A modular and flexible library designed around a unifying computational graph to empower users with composable deep learning abstractions.

2. Execution of these models with maximal computational efficiency without worrying about details such as kernel fusion/compounding or data layout.

3. Enabling all of this on any user's hardware whether one or multiple CPUs, GPUs, and/or Nervana Engines.

To achieve these goals, the ngraph library has three layers:

1. An API for creating computational ngraphs.

2. Two higher level frontend APIs (TensorFlow and Neon) utilizing the ngraph API for common deep learning workflows.

3. A transformer API for compiling these graphs and executing them on GPUs and CPUs.

.. image :: assets/ngraph_workflow.png

Let us consider each of these in turn and the way they enable users.

Nervana Graphs
--------------
The computational graphs of Theano and |TF| require a user to reason about the underlying tensor shapes while constructing the graph. This is tedious and error prone for the user and eliminates the ability for a compiler to reorder axes to match the assumptions of particular hardware platforms as well.

To simplify tensor management, the ngraph API enables users to define a set of named axes, attach them to tensors during graph construction, and specify them by name (rather than position) when needed.  These axes can be named according to the particular domain of the problem at hand to help a user with these tasks.  This flexibility then allows the necessary reshaping/shuffling to be inferred by the transformer before execution. Additionally, these inferred tensor axis orderings can then be optimized across the entire computational graph for ordering preferences of the underlying runtimes/hardware platforms to optimize cache locality and runtime execution time.

These capabilities highlight one of the tenants of ngraph, which is to operate at a higher level of abstraction so transformers can make execution efficient without needing a "sufficiently smart compiler" that can reverse-engineer the higher level structure, as well as allowing users and frontends to more easily compose these building blocks together.

Frontends
---------
Most applications and users don't need the full flexibility offered by the ngraph API, so we are also introducing a higher level ``neon`` API which offers a user a composable interface with the common building blocks to construct deep learning models. This includes things like common optimizers, metrics, and layer types such as linear, batch norm, convolutional, and RNN. We also illustrate these with example networks training on MNIST digits, CIFAR10 images, and the Penn Treebank text corpus.

We also realize that users already know and use existing frameworks today and might want to continue using/combine models written in other frameworks. To that end, we demonstrate the capability to **convert existing tensorflow models into ngraphs** and execute them using ngraph transformers. This importer supports a variety of common operation types today and will be expanding in future releases. We also plan on implementing compatibility with other frameworks in the near future, so stay tuned.

Additionally, we wish to stress that because ngraph offers the core building blocks of deep learning computation and multiple high performance backends, adding frontends is a straightforward affair and improvements to a backend (or new backends) are automatically leveraged by all existing and future frontends. So users get to keep using their preferred syntax while benefiting from the shared compilation machinery.

Transformers
------------
Making sure that models execute quickly with minimal memory overhead is critical given the millions or even billions of parameters and weeks of training time used by state of the art models. Given our experience building and maintaining the fastest deep learning library on GPUs, we appreciate the complexities of modern deep learning performance:

- Kernel fusion/compounding
- Efficient buffer allocation
- Training vs. inference optimizations
- Heterogeneous backends
- Distributed training
- Multiple data layouts
- New hardware advancements (eg: Nervana Engine)

With these realities in mind, we designed ngraph transformers to automate and abstract these details away from frontends through clean APIs, while allowing the power user room to tweak things all simultaneously while not limiting the flexible abstractions for model creation.  In ngraph, we believe the key to achieving these goals rests in standing on the shoulders of giants in `modern compiler design <http://www.aosabook.org/en/llvm.html>`_ to promote flexibility and experimentation in choosing the set and order of compiler optimizations for a transformer to use.

Each ngraph transformer (or backend in LLVM parlance) targets a particular hardware backend and acts as an interface to compile an ngraph into a computation that is ready to be evaluated by the user as a function handle.

Today ngraph ships with a transformer for GPU and CPU execution, but in the future we plan on implementing heterogeneous device transformers with distributed training support.

Example
-------
For an example of building and executing ngraphs, please see the :doc:`walkthrough<walk_throughs>` in our documentation, but we include here a "hello world" example, which will print the numbers ``1`` through ``5``.

.. code:: python

    import ngraph as ng
    import ngraph.transformers as ngt

    # Build a graph
    x = ng.placeholder(())
    x_plus_one = x + 1

    # Construct a transformer
    transformer = ngt.make_transformer()

    # Define a computation
    plus_one = transformer.computation(x_plus_one, x)

    # Run the computation
    for i in range(5):
        print(plus_one(i))

Status and Future Work
----------------------

As this is a preview release, we have much work left to do. Currently we include working examples
of:

- MLP networks using MNIST and CIFAR-10.
- Convolutional networks using MNIST and CIFAR-10.
- RNN's using Penn Treebank.

We are actively working towards:

- Graph serialization/deserialization.
- Further improvements to graph composability for usability/optimization.
- Add additional support for more popular frontends.
- Distributed, heterogeneous backend target support.
- C APIs for interoperability to enable other languages to create/execute graphs.
- Modern, cloud native model deployment strategies.
- Reinforcement learning friendly `network construction <http://openreview.net/forum?id=r1Ue8Hcxg>`_ frontends.

Join us
-------
With the rapid pace in the deep learning community we realize that a project like this won't succeed without community participation, which is what motivated us to put this preview release out there to get feedback and encourage people like you to come join us in defining the next wave of deep learning tooling. Feel free to make pull requests/suggestions/comments on `Github <https://github.com/NervanaSystems/ngraph>`_) or reach out to us on our `mailing list <https://groups.google.com/forum/#!forum/neon-users>`_. We are also hiring for full-time and internship positions.