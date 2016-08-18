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
import collections

from geon.frontends.neon.layer import Layer, BranchNode


def flatten(item):
    """
    TODO.

    Arguments:
      item: TODO

    Returns:

    """
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

    Arguments:

    Returns:

    """

    def __init__(self, **kargs):
        super(LayerContainer, self).__init__(**kargs)


class Sequential(LayerContainer):
    """
    Layer container that encapsulates a simple linear pathway of layers.

    Arguments:
      layers: list
      including: layer containers

    Returns:

    """

    def __init__(self, layers, **kargs):
        super(Sequential, self).__init__(**kargs)
        self.layers = [l for l in flatten(layers)]
        self._layers = [x for x in self.layers if type(x) not in (BranchNode,)]
#       root = self._layers[0]

    def configure(self, in_obj):
        """
        Must receive a list of shapes for configuration (one for each pathway)
        the shapes correspond to the layer_container attribute

        Arguments:
          in_obj: any object that has an out_shape

        Returns:

        """
        config_layers = self.layers if in_obj else self._layers
        in_obj = in_obj if in_obj else self.layers[0]
        in_obj = super(Sequential, self).configure(in_obj)
        for l in config_layers:
            in_obj = l.configure(in_obj)
        return in_obj


class Tree(LayerContainer):
    """
    Layer container that encapsulates a simple linear pathway of layers.

    Arguments:
      layers: list
      The: branches must be provided with main trunk first
      branches: in the order the branch nodes are encountered
      name: string
      alphas: list
      backpropagating: error

    Returns:

    """

    def __init__(self, layers, alphas=None, **kargs):
        super(Tree, self).__init__(**kargs)


class SingleOutputTree(Tree):
    """
    Subclass of the Tree container which returns only
    the output of the main branch (branch index 0) during
    inference.

    Arguments:

    Returns:

    """

    def __init__(self, **kargs):
        super(SingleOutputTree, self).__init__(**kargs)
