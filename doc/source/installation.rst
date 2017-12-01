.. _installation:

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

Installation
############

Requirements
============

Intel® Nervana™ Graph requires **Python 2.7** or **Python 3.4+** running on a 
Linux* or UNIX-based OS. Before installing, also ensure your system has recent 
updates of the following packages:

.. csv-table::
   :header: "Ubuntu* 16.04+ or CentOS* 7.4+", "Mac OS X*", "Description"
   :widths: 20, 20, 42
   :escape: ~

   python-pip, pip, Tool to install Python dependencies
   python-virtualenv (*), virtualenv (*), Allows creation of isolated environments ((*): This is required only for Python 2.7 installs. With Python3: test for presence of ``venv`` with ``python3 -m venv -h``)
   libhdf5-dev, h5py, Enables loading of hdf5 formats
   libyaml-dev, pyaml, Parses YAML format inputs
   pkg-config, pkg-config, Retrieves information about installed libraries


Prerequisites  
=============

#. **Choose your build environment.** Installing within a virtual environment
   is the easiest option for most users. To prepare for a system installation,
   you may skip this step.  

   * **Python3** 
     To create and activate a Python 3 virtualenv:
     
    .. code-block:: console
   
       $ python3 -m venv .venv
       $ . .venv/bin/activate

   * **Python 2.7**
     To create and activate a Python 2 virtualenv:

    .. code-block:: console

       $ virtualenv -p python2.7 .venv
       $ . .venv/bin/activate

#. **Download the source code.**

    .. code-block:: console

       $ git clone git@github.com:NervanaSystems/ngraph.git
       $ cd ngraph

#. **ONNX dependency.**  

   * To build with the `ONNX`_ dependency, extra steps to support ONNX are
     needed. For more information about Nervana Graph support for ONNX, you
     can read `this blog post`_. The following commands will compile ONNX 
     on CentOS v7.4+ systems:

     .. code-block:: console

        $ yum install autoconf automake libtool curl g++ unzip -y
        $ git clone https://github.com/google/protobuf.git
        $ cd protobuf
        $ ./autogen.sh
        $ ./configure
        $ make && make install
        $ ldconfig

   * To prepare the ONNX dependency on Mac OS X* or Ubuntu* systems, run:
       
     .. code-block:: console

        $ make onnx_dependency



Installation
============
  
To build and install Intel Nervana Graph, simply run ``make install`` from within the
clone of the repo as follows:

.. code-block:: console

   $ make install



Back-end Configuration
======================

After completing the prerequisites and installation of the base Nervana
Graph package, additional packages can be added to achieve optimal
performance when running on your various backend platforms.

#. **CPU/Intel® architecture transformer**
   
   (Optional) To run Intel Nervana Graph with optimal performance on a CPU
   backend, configure your build of Nervana Graph with the Intel® Math Kernel 
   Library for Deep Neural Networks, AKA the Intel® `MKL DNN`_, a new open-source 
   library designed to accelerate Deep Learning (DL) applications on Intel® 
   architecture.

   .. code-block:: console

      $ git clone https://github.com/01org/mkl-dnn.git
      $ cd mkl-dnn/scripts && ./prepare_mkl.sh && cd ..
      $ mkdir -p build && cd build
      $ cmake -DCMAKE_INSTALL_PREFIX=$PWD/../install .. && make install
      $ cd ../.. && export MKLDNN_ROOT=$PWD/mkl-dnn/install

#. **GPU transformer**

   (Optional) Enabling neon to use GPUs requires installation of 
   `CUDA SDK and drivers`_. Remember to add the CUDA path to your 
   environment variables:
  
  * On Ubuntu
  
    .. code-block:: bash

       export PATH="/usr/local/cuda/bin:"$PATH
       export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda/lib:/usr/local/lib:"$LD_LIBRARY_PATH

  * On Mac OS X

    .. code-block:: bash

       export PATH="/usr/local/cuda/bin:"$PATH
       export DYLD_LIBRARY_PATH="/usr/local/cuda/lib:"$DYLD_LIBRARY_PATH

  * To add GPU support after installing, you can also run:

    .. code-block:: console

       $ make gpu_prepare


Getting Started
===============

Some Jupyter* notebook walkthroughs demonstrate ways to use Intel Nervana Graph:

* ``examples/walk_through/``: Use Nervana Graph to implement logistic regression 
* ``examples/mnist/MNIST_Direct.ipynb``: Build a deep learning model directly on 
  Nervana Graph

The `neon framework`_ can also be used to define and train deep learning models:

* ``examples/mnist/mnist_mlp.py``: Multilayer perceptron network on MNIST dataset.
* ``examples/cifar10/cifar10_conv.py``: Convolutional neural network on CIFAR-10.
* ``examples/cifar10/cifar10_mlp.py``: Multilayer perceptron on CIFAR-10 dataset.
* ``examples/ptb/char_rnn.py``: Character-level RNN model on Penn Treebank data.

Some TensorFlow* examples that define graphs which can be passed to ngraph for 
execution are also included:

* ``frontends/tensorflow/examples/minimal.py``
* ``frontends/tensorflow/examples/logistic_regression.py``
* ``frontends/tensorflow/examples/mnist_mlp.py``


Developer Guidelines
====================

Before checking in code, run the unit tests and check for style errors:

.. code-block:: console

   $ make test_cpu test_gpu test_integration
   $ make style

Documentation can be generated with pandoc:

.. code-block:: console

   $ sudo apt-get install pandoc
   $ make doc

View the documentation at ``doc/build/html/index.html``.


.. _neon framework: http://neon.nervanasys.com/index.html/installation.html
.. _CUDA SDK and drivers: https://developer.nvidia.com/cuda-downloads
.. _OpenBLAS: http://www.openblas.net
.. _see sample instructions here: https://hunseblog.wordpress.com/2014/09/15/installing-numpy-and-openblas
.. _this blog post: https://www.intelnervana.com/intel-joins-open-neural-network-exchange-ecosystem
.. _Pascal: http://developer.nvidia.com/pascal
.. _Maxwell: http://maxwell.nvidia.com
.. _Kepler: http://www.nvidia.com/object/nvidia-kepler.html
.. _MKL DNN: https://github.com/01org/mkl-dnn/
.. _ONNX: http://onnx.ai/
.. _neon: http://neon.nervanasys.com/index.html

