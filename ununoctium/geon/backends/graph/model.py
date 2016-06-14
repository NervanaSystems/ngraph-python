import numpy as np

from geon.backends.graph.names import NameableValue, bound_naming
from geon.backends.graph.environment import bound_environment, captured_ops
from geon.backends.graph.graph import GraphComponent
from geon.backends.graph.arrayaxes import AxisVar
from geon.backends.graph.container import Sequential, Tree, SingleOutputTree



class Model(object, NameableValue, GraphComponent):
    def __init__(self, layers, optimizer=None, **kargs):
        super(Model, self).__init__(**kargs)
        self.initialized = False

        self.optimizer = optimizer

        graph = self.graph
        # Define the standar Neon axes
        graph.N = AxisVar()
        graph.C = AxisVar()
        graph.D = AxisVar()
        graph.H = AxisVar()
        graph.W = AxisVar()
        graph.T = AxisVar()
        graph.R = AxisVar()
        graph.S = AxisVar()
        graph.K = AxisVar()
        graph.M = AxisVar()
        graph.P = AxisVar()
        graph.Q = AxisVar()

        # Wrap the list of layers in a Sequential container if a raw list of layers
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
        output = self.layers.configure(self.graph, dataset)

        if cost is not None:
            cost.initialize(output)
            self.cost = cost

        self.initialized = True


    def fit(self, dataset, cost, optimizer, num_epochs, callbacks):
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

        with bound_environment(environmant=self.environment):
            with bound_naming(naming=self.graph):
                with captured_ops as self.ops:
                    self.initialize(dataset, cost)

        callbacks.on_train_begin(num_epochs)
        while self.epoch_index < num_epochs and not self.finished:
            self.nbatches = dataset.nbatches

            callbacks.on_epoch_begin(self.epoch_index)

            self._epoch_fit(dataset, callbacks)

            callbacks.on_epoch_end(self.epoch_index)

            self.epoch_index += 1

        callbacks.on_train_end()

