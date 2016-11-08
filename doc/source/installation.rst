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


Installation
************

To install |geon|, be sure to first install neon in a virtual environment.
Activate the virtualenv with ``. .venv/bin/activate``, then run::

    git clone git@github.com:NervanaSystems/ngraph.git
    cd ngraph
    make install

Examples
========

Several useful example scripts demonstate how to use Nervana Graph:

* ``ngraph/examples/walk_through/`` contains several walkthrough code.
* ``ngraph/examples/mnist_mlp.py`` uses the neon front-end to define and train the model.
* ``ngraph/examples/cifar10_mlp.py`` uses the neon front-end to define and train the model.

Developer Guide
===============

Before checking in code, run the unit tests and check for style errors::

    make test
    make style

Documentation can be generated via::

    make doc


The latest html documentation is also built by Jenkins and can be viewed
`here <http://jenkins.sd.nervana.ai:8080/job/NEON_NGRAPH_Integration_Test/lastSuccessfulBuild/artifact/doc/build/html/index.html>`_.
