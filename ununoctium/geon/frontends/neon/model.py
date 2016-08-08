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

import geon.frontends.base.axis as ax
import geon as be
import geon.transformers.nptransform as nptransform
from geon.backends.graph.environment import bound_environment
from geon.frontends.base.graph import GraphComponent
from geon.frontends.neon.container import Sequential, Tree, SingleOutputTree
from geon.op_graph.names import name_scope


def dataset_nclasses(dataset):
    if isinstance(dataset, ArrayIterator):
        return dataset.nclass
    elif isinstance(dataset, DataLoader):
        return dataset.nclasses


def dataset_batchsize(dataset):
    if isinstance(dataset, ArrayIterator):
        return dataset.be.bsz
    elif isinstance(dataset, DataLoader):
        return dataset.bsz


class Model(GraphComponent):

    def __init__(self, layers, name=None, optimizer=None, **kargs):
        super(Model, self).__init__(**kargs)
        self.initialized = False
        self.name = name
        self.epoch_index = 0
        self.finished = False

        self.optimizer = optimizer
        # Wrap the list of layers in a Sequential container if a raw list of
        # layers
        if type(layers) in (Sequential, Tree, SingleOutputTree):
            self.layers = layers
        else:
            self.layers = Sequential(layers)

    def initialize(self, dataset, cost=None):
        """
        Propagate shapes through the layers to configure, then allocate space.

        Arguments:
            dataset (NervanaDataIterator): Dataset iterator to perform initialization on
            cost (Cost): Defines the function which the model is minimizing based
                         on the output of the last layer and the input labels.
        """
        if self.initialized:
            return

        # Propagate shapes through the layers to configure
        self.output = self.layers.configure(dataset)

        self.cost = cost
        if cost is not None:
            self.cost.initialize(self.output, self.target)

        self.initialized = True
        return self.output

    def fit(
            self,
            dataset,
            input_axes,
            target_axes,
            cost,
            optimizer,
            num_epochs,
            callbacks):
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
            optimizer (Optimizer): Defines the learning rule for updating the model parameters.
            num_epochs: Number of times to iterate over the dataset.
            callbacks (Callbacks): Defines callbacks to run at the end of each mini-batch / epoch.
        """
        self.nbatches = dataset.nbatches
        self.ndata = dataset.ndata
        self.optimizer = optimizer

        with bound_environment(environment=self.environment):
            with name_scope(name_scope=self.graph):
                # TODO Move this axis initialization into a util
                batch_input_axes = input_axes + be.Axes(ax.N, )
                batch_target_axes = target_axes + be.Axes(ax.N, )
                be.set_batch_axes(be.Axes(ax.N, ))
                be.set_phase_axes(be.Axes(ax.Phi, ))
                self.input = be.placeholder(axes=batch_input_axes)
                self.target = be.placeholder(axes=batch_target_axes)
                for axis, length in zip(input_axes, dataset.shape):
                    axis.length = length
                for axis, length in zip(
                    target_axes, [
                        dataset_nclasses(dataset)]):
                    axis.length = length
                ax.N.length = dataset_batchsize(dataset)
                ax.Phi.length = 2
                self.batch_input_shape = batch_input_axes.lengths
                self.batch_target_shape = batch_target_axes.lengths

                self.initialize(self.input, cost)
                updates = self.optimizer.configure(self.cost.total_cost)

                self.enp = be.NumPyTransformer(
                    results=[self.cost.mean_cost, updates])

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
            dataset (NervanaDataIterator): Dataset iterator to perform fit on
        """
        epoch = self.epoch_index
        self.total_cost = 0
        batch = 0
        # iterate through minibatches of the dataset
        for mb_idx, (x, t) in enumerate(dataset):
            callbacks.on_minibatch_begin(epoch, mb_idx)
            self.input.value = x.reshape(self.batch_input_shape)
            self.target.value = t.reshape(self.batch_target_shape)
            self.optimizer.optimize(self.epoch_index)

            vals = self.enp.evaluate()
            batch_cost = vals[self.cost.mean_cost]
            self.cost.cost = batch_cost
            self.total_cost += batch_cost
            batch = batch + 1
            callbacks.on_minibatch_end(epoch, mb_idx)

        # now we divide total cost by the number of batches,
        # so it was never total cost, but sum of averages
        # across all the minibatches we trained on
        self.total_cost = self.total_cost / dataset.nbatches

    @be.with_graph_scope
    def epoch_eval(self, dataset):
        with be.bound_environment():
            nprocessed = 0
            self.loss = 0
            dataset.reset()
            enp = nptransform.NumPyTransformer(results=[self.cost.mean_cost])
            for x, t in dataset:
                self.input.value = x.reshape(self.batch_input_shape)
                self.target.value = t.reshape(self.batch_target_shape)
                bsz = min(dataset.ndata - nprocessed, dataset_batchsize(dataset))
                nsteps = x.shape[1] // dataset_batchsize(dataset) if not isinstance(
                    x, list) else x[0].shape[1] // dataset_batchsize(dataset)
                vals = enp.evaluate()
                batch_cost = vals[self.cost.mean_cost]
                nprocessed += bsz
                self.loss += batch_cost / nsteps
            return float(self.loss) / nprocessed

    @be.with_graph_scope
    def eval(self, dataset, metric):
        """
        Evaluates a model on a dataset according to an input metric.

        Arguments:
            datasets (NervanaDataIterator): dataset to evaluate on.
            metric (Cost): what function to evaluate dataset on.

        Returns:
            Host numpy array: the error of the final layer for the evaluation dataset
        """
        self.initialize(dataset)
        with be.bound_environment(self.environment):
            running_error = np.zeros(
                (len(metric.metric_names)), dtype=np.float32)
            nprocessed = 0
            dataset.reset()
            error = metric(self.output, self.target)
            enp = nptransform.NumPyTransformer(results=[error])
            for x, t in dataset:
                self.input.value = x.reshape(self.batch_input_shape)
                self.target.value = t.reshape(self.batch_target_shape)
                bsz = min(dataset.ndata - nprocessed,
                          dataset_batchsize(dataset))
                nsteps = x.shape[1] // dataset_batchsize(dataset) if not isinstance(
                    x, list) else x[0].shape[1] // dataset_batchsize(dataset)
                # calcrange = slice(0, nsteps * bsz)
                vals = enp.evaluate()
                error_val = vals[error]
                running_error += error_val * bsz * nsteps
                nprocessed += bsz * nsteps
            running_error /= nprocessed
            return running_error

    def serialize(self, fn=None, keep_states=True):
        # TODO
        pass
