.. _neon:

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

neon™
*****

The neon™ frontend to Intel® Nervana™ graph (ngraph) provides common deep learning primitives, such as activation functions, optimizers, layers, and more. We include several examples in this release to illustrate how to use the neon frontend to construct your models:

- ``examples/minst/mnist_mlp.py``: Multi-layer perceptron on the MNIST digits dataset.
- ``examples/cifar10/cifar10_mlp.py``: Multi-layer perceptron on the CIFAR10 dataset.
- ``examples/cifar10/cifar10_conv.py``: Convolutional neural networks applied to the CIFAR10 dataset.
- ``examples/ptb/char_rnn.py``: Character-level RNN language model on the Penn Treebank dataset.

We currently have support for the following sets of deep learning primitives:

- Layers: ``Linear``, ``Bias``, ``Conv2D``, ``Pool2D``, ``BatchNorm``, ``Dropout``, ``Recurrent``
- Activations: ``Rectlin``, ``Rectlinclip``, ``Identity``, ``Explin``, ``Normalizer``, ``Softmax``, ``Tanh``, ``Logistic``
- Initializers: ``GaussianInit``, ``UniformInit``, ``ConstantInit``
- Optimizers: ``GradientDescentMomentum``, ``RMSprop``
- Callbacks: ``TrainCostCallback``, ``RunTimerCallback``, ``ProgressCallback``, ``TrainLoggerCallback``, ``LossCallback``


MNIST Example
-------------

The ``mnist_mlp.py`` example provides an introduction into the neon frontend. Similiar to ``neon``, we begin with an arg parser that allows command line flags for specifying options, such as batch size and the data directory:

.. code-block:: python

	from ngraph.frontends.neon import NgraphArgparser
	parser = NgraphArgparser(description='Train simple mlp on mnist dataset')
	args = parser.parse_args()

To provision data to the model, we use the ``ArrayIterator`` object, which is a Python iterator that returns a minibatch of inputs and targets to the model with each call. We also provide a helper function for handling the MNIST dataset and providing NumPY arrays with the image and target data.

.. code-block:: python

	from mnist import MNIST
	from ngraph.frontends.neon import ArrayIterator

	# Create the dataloader
	train_data, valid_data = MNIST(args.data_dir).load_data()
	train_set = ArrayIterator(train_data, args.batch_size, total_iterations=args.num_iterations)
	valid_set = ArrayIterator(valid_data, args.batch_size)

You can compose models as a list of layers which then gets passed to a container that handles how the layers are linked. Here we just have a sequential list of layers, so we use the ``Sequential`` container.

.. code-block:: python

	from ngraph.frontends.neon import Affine, Preprocess, Sequential
	from ngraph.frontends.neon import GaussianInit, Rectlin, Logistic, GradientDescentMomentum

	seq1 = Sequential([Preprocess(functor=lambda x: x / 255.),
	                   Affine(nout=100, weight_init=GaussianInit(), activation=Rectlin()),
	                   Affine(axes=ax.Y, weight_init=GaussianInit(), activation=Logistic())])

Note that above we also use a ``Preprocess`` layer to scale the input image data to between 0 and 1.

The ``neon`` frontend has a predefined set of axes commonly used with deep learning in the ``ngraph/frontends/neon/axis.py`` file. We import the *name_scope* from this file as ``ax``, and can define the lengths of the relevant axes give the shape of the input image and the target number of classes:

.. code-block:: python

	from ngraph.frontends.neon import ax

	ax.C.length, ax.H.length, ax.W.length = train_set.shapes['image']
	ax.N.length = args.batch_size
	ax.Y.length = 10

With these axes defined, we can then define placeholders for the inputs. Our image has axes ``CHWN``, and the target label has axes ``N`` for the batch size.

.. code-block:: python

	inputs = dict(image=ng.placeholder([ax.C, ax.H, ax.W, ax.N]),
	              label=ng.placeholder([ax.N]))

We use stochastic gradient descent with momentum.

.. code-block:: python

	optimizer = GradientDescentMomentum(0.1, 0.9)

We then define the model output, and the associated cost function and metric (the misclassification rate) using the ngraph API directly:

.. code-block:: python

	output_prob = seq1.train_outputs(inputs['image'])

	errors = ng.not_equal(ng.argmax(output_prob, out_axes=[ax.N]), inputs['label'])
	loss = ng.cross_entropy_binary(output_prob, ng.one_hot(inputs['label'], axis=ax.Y))

	mean_cost = ng.mean(loss, out_axes=())
	updates = optimizer(loss)

To obtain the model output, we use the sequential container's included `train_outputs()` method, which essentially performs the forward pass through the layers of the model.

Now that we have used the neon frontend to compose our graph, we pass it to a transformer for execution by specifying the computations required to both train the network and also to compute the loss. Instead of directly specifying the computations using ``transformer.computation()`` as with the Intel Nervana graph walkthrough examples, we instead use a helper function ``make_bound_computation()`` to create computations that bind a set of inputs with outputs. We can specify a set of outputs using Python dictionaries.

.. code-block:: python

	from ngraph.frontends.neon import make_bound_computation, make_default_callbacks

	train_outputs = dict(batch_cost=mean_cost, updates=updates)
	loss_outputs = dict(cross_ent_loss=loss, misclass_pct=errors)

	# Now bind the computations we are interested in
	transformer = ngt.make_transformer()
	train_computation = make_bound_computation(transformer, train_outputs, inputs)
	loss_computation = make_bound_computation(transformer, loss_outputs, inputs)

In the case of ``train_computation``, we can think of ``make_bound_computation`` as creating a computation by calling ``transformer.computation([mean_cost updates], inputs)``.

Callbacks allow the model to report back its progress and any relevant metrics during the course of training.

.. code-block:: python

	from ngraph.frontends.neon import make_default_callbacks

	cbs = make_default_callbacks(transformer=transformer,
	                         output_file=args.output_file,
                             frequency=args.iter_interval,
                             train_computation=train_computation,
                             total_iterations=args.num_iterations,
                             eval_set=valid_set,
                             loss_computation=loss_computation,
                             use_progress_bar=args.progress_bar)

Finally, we use another helper function, ``loop_train``, to train the model. ``loop_train`` loops through the provided training data, calling the provided computation (in this case ``train_computation``), to update the model weights and report progress via the provided callbacks.

.. code-block:: python

	from ngraph.frontends.neon import loop_train

	loop_train(train_set, train_computation, cbs)

.. Note::
   This model is very similar to the ``MNIST_Direct.ipynb``, which walks through an implementation using the Intel Nervana graph API directly instead of the neon frontend. The neon frontend essentially contains objects and helper methods that wrap the ngraph calls to make it easier for users to compose the networks in terms of deep learning building blocks.