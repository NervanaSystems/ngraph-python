# ----------------------------------------------------------------------------
# Copyright 2016 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
from __future__ import division

import numpy as np
from builtins import zip
from neon.data import ArrayIterator, DataLoader

import ngraph.frontends.base.axis as ax
import ngraph as ng
from ngraph.frontends.neon.container import Sequential, Tree, SingleOutputTree


def dataset_nclasses(dataset):
    """
    TODO.

    Arguments:
      dataset: TODO

    Returns:

    """
    if isinstance(dataset, ArrayIterator):
        return dataset.nclass
    elif isinstance(dataset, DataLoader):
        return dataset.nclasses


def dataset_batchsize(dataset):
    """
    TODO.

    Arguments:
      dataset: TODO

    Returns:

    """
    if isinstance(dataset, ArrayIterator):
        return dataset.be.bsz
    elif isinstance(dataset, DataLoader):
        return dataset.bsz


class Model(object):
    """TODO."""

    def __init__(self, layers, name=None, optimizer=None, **kwargs):
        super(Model, self).__init__(**kwargs)
        self.initialized = False
        self.name = name
        self.epoch_index = 0
        self.finished = False

        # Wrap the list of layers in a Sequential container if a raw list of
        # layers
        if type(layers) in (Sequential, Tree, SingleOutputTree):
            self.layers = layers
        else:
            self.layers = Sequential(layers)

        self.transformer = None
        self.train_comp = None
        self.test_comp = None
        self.metric = None
        self.cost = None

    def initialize(self,
                   dataset, input_axes, target_axes,
                   cost, optimizer, metric=None):
        """
        Propagate shapes through the layers to configure, then allocate space.

        Arguments:
           dataset (NervanaDataIterator): An iterable of minibatches where each
               element is a (x, y) tuple where x is the input data and y are the labels.
               x is of dimension (feature_size, batch_size)
               y is of dimension (label_size, batch_size)
               Length of the iterator is num_batches which is num_data / batch_size.
           cost (Cost): Defines the function which the model is minimizing based
                        on the output of the last layer and the input labels.
           optimizer (Optimizer): Defines the learning rule for updating the model parameters.
           num_epochs: Number of times to iterate over the dataset.
           callbacks (Callbacks): Defines callbacks to run at the end of each mini-batch / epoch.

        Returns:

        """
        if self.initialized:
            return

        self.optimizer = optimizer

        batch_input_axes = input_axes + ng.Axes(ax.N, )
        batch_target_axes = target_axes + ng.Axes(ax.N, )
        self.input = ng.placeholder(axes=batch_input_axes)
        self.target = ng.placeholder(axes=batch_target_axes)
        for axis, length in zip(input_axes, dataset.shape):
            axis.length = length
        for axis, length in zip(
            target_axes, [
                dataset_nclasses(dataset)]):
            axis.length = length
        ax.N.length = dataset_batchsize(dataset)
        self.batch_input_shape = batch_input_axes.lengths
        self.batch_target_shape = batch_target_axes.lengths

        # Propagate shapes through the layers to configure
        self.output = self.layers.configure(self.input)

        self.cost = cost
        self.cost.initialize(self.output, self.target)
        self.transformer = ng.NumPyTransformer()
        updates = self.optimizer.configure(
            self.transformer,
            self.cost.mean_cost
        )

        self.train_comp = self.transformer.computation([self.cost.mean_cost, updates], self.input,
                                                       self.target)
        self.epoch_eval_comp = self.transformer.computation(self.cost.mean_cost, self.input,
                                                            self.target)

        if metric is not None:
            self.metric = metric
            self.error = metric(self.output, self.target)
            self.test_comp = self.transformer.computation(self.error, self.input, self.target)

        self.transformer.initialize()
        self.initialized = True

    def fit(self, dataset, num_epochs, callbacks):
        """
        Trains the model parameters on a dataset by minimizing the cost function through
        gradient descent and updates the layer weights according to a learning rule
        defined in optimizer.

        Arguments:
           dataset (NervanaDataIterator): An iterable of minibatches where each
               element is a (x, y) tuple where x is the input data and y are the labels.
               x is of dimension (feature_size, batch_size)
               y is of dimension (label_size, batch_size)
               Length of the iterator is num_batches which is num_data / batch_size.
           cost (Cost): Defines the function which the model is minimizing based
                        on the output of the last layer and the input labels.
           num_epochs: Number of times to iterate over the dataset.
           callbacks (Callbacks): Defines callbacks to run at the end of each mini-batch / epoch.

        Returns:

        """
        self.nbatches = dataset.nbatches
        self.ndata = dataset.ndata
        callbacks.on_train_begin(num_epochs)
        while self.epoch_index < num_epochs and not self.finished:
            self.nbatches = dataset.nbatches
            callbacks.on_epoch_begin(self.epoch_index)
            self._epoch_fit(dataset, callbacks)
            callbacks.on_epoch_end(self.epoch_index)
            self.epoch_index += 1
        callbacks.on_train_end()

    def _epoch_fit(self, dataset, callbacks):
        """
        Helper function for fit which performs training on a dataset for one epoch.

        Arguments:
          dataset (NervanaDataIterator): Dataset iterator to perform fit on.
          callbacks: TODO

        Returns:

        """
        epoch = self.epoch_index
        self.total_cost = 0
        batch = 0
        # iterate through minibatches of the dataset
        for mb_idx, (x, t) in enumerate(dataset):
            callbacks.on_minibatch_begin(epoch, mb_idx)
            self.optimizer.optimize(self.epoch_index)

            batch_cost, _ = self.train_comp(x.reshape(self.batch_input_shape),
                                            t.reshape(self.batch_target_shape))
            self.cost.cost = batch_cost
            self.total_cost += batch_cost
            batch = batch + 1
            callbacks.on_minibatch_end(epoch, mb_idx)

        # now we divide total cost by the number of batches,
        # so it was never total cost, but sum of averages
        # across all the minibatches we trained on
        self.total_cost = self.total_cost / dataset.nbatches

    def epoch_eval(self, dataset):
        """
        TODO.

        Arguments:
          dataset: TODO

        Returns:

        """
        nprocessed = 0
        self.loss = 0
        dataset.reset()
        for x, t in dataset:
            bsz = min(dataset.ndata - nprocessed, dataset_batchsize(dataset))
            nsteps = x.shape[1] // dataset_batchsize(dataset)\
                if not isinstance(x, list)\
                else x[0].shape[1] // dataset_batchsize(dataset)
            batch_cost = self.epoch_eval_comp(x.reshape(self.batch_input_shape),
                                              t.reshape(self.batch_target_shape))
            nprocessed += bsz
            self.loss += batch_cost / nsteps
        return float(self.loss) / nprocessed

    def eval(self, dataset):
        """
        Evaluates a model on a dataset according to an input metric.

        Arguments:
          datasets (NervanaDataIterator): dataset to evaluate on.
          metric (Cost): what function to evaluate dataset on.

        Returns:
          Host numpy array: the error of the final layer for the evaluation dataset

        """
        running_error = np.zeros(
            (len(self.metric.metric_names)), dtype=np.float32)
        nprocessed = 0
        dataset.reset()
        for x, t in dataset:
            bsz = min(dataset.ndata - nprocessed,
                      dataset_batchsize(dataset))
            nsteps = x.shape[1] // dataset_batchsize(dataset)\
                if not isinstance(x, list)\
                else x[0].shape[1] // dataset_batchsize(dataset)
            # calcrange = slice(0, nsteps * bsz)
            error_val = self.test_comp(x.reshape(self.batch_input_shape),
                                       t.reshape(self.batch_target_shape))
            running_error += error_val * bsz * nsteps
            nprocessed += bsz * nsteps
        running_error /= nprocessed
        return running_error

    def serialize(self, fn=None, keep_states=True):
        """
        TODO.

        Arguments:
          fn: TODO
          keep_states: TODO

        Returns:

        """
        # TODO
        pass
