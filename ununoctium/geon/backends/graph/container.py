import collections

from neon.layers.container import flatten

from geon.backends.graph.names import NameScope
from geon.backends.graph.layer import Layer, BranchNode


def flatten(item):
    if isinstance(item, collections.Iterable):
        for i in iter(item):
            for j in flatten(i):
                yield j
    else:
        yield item


class LayerContainer(Layer):
    """
    Layer containers are a generic class that are used to encapsulate groups of layers and
    provide methods for propagating through the constituent layers, allocating memory.
    """
    def __init__(self, **kargs):
        super(LayerContainer, self).__init__(**kargs)


class Sequential(LayerContainer):
    """
    Layer container that encapsulates a simple linear pathway of layers.

    Arguments:
        layers (list): List of objects which can be either a list of layers
                       (including layer containers).
    """
    def __init__(self, layers, **kargs):
        super(Sequential, self).__init__(**kargs)
        self.layers = [l for l in flatten(layers)]
        self._layers = [x for x in self.layers if type(x) not in (BranchNode,)]
        root = self._layers[0]


    def configure(self, graph, in_obj):
        """
        Must receive a list of shapes for configuration (one for each pathway)
        the shapes correspond to the layer_container attribute

        Arguments:
            in_obj: any object that has an out_shape (Layer) or shape (Tensor, dataset)
        """
        config_layers = self.layers if in_obj else self._layers
        in_obj = in_obj if in_obj else self.layers[0]
        in_obj = super(Sequential, self).configure(graph, in_obj)
        for l in config_layers:
            in_obj = l.configure(graph, in_obj)
        return in_obj



class Tree(LayerContainer):
    """
    Layer container that encapsulates a simple linear pathway of layers.

    Arguments:
        layers (list): List of Sequential containers corresponding to the branches of the Tree.
                       The branches must be provided with main trunk first, and then the auxiliary
                       branches in the order the branch nodes are encountered
        name (string, optional): Name for the container
        alphas (list(float), optional): list of weighting factors to apply to each branch for
                                        backpropagating error.
    """

    def __init__(self, layers, alphas=None, **kargs):
        super(Tree, self).__init__(**kargs)


class SingleOutputTree(Tree):
    """
    Subclass of the Tree container which returns only
    the output of the main branch (branch index 0) during
    inference.
    """
    def __init__(self, **kargs):
        super(SingleOutputTree, self).__init__(**kargs)
