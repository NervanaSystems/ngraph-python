# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 01:04:05 2017

@author: mphielipp
"""

import ngraph as ng
from ngraph.frontends.neon import LookupTable
from ngraph.frontends.neon import UniformInit
from ngraph.frontends.neon import Affine, Sequential, XavierInit
from ngraph.frontends.neon import Dropout
from ngraph.frontends.neon import SubGraph


# Please read data.py to understand how the data is being preprocess.

print("Creating the Deep Learning model...")


class WideDeepClassifier(SubGraph):

    """
    Implementation of the Wide and Deep model.

    Reference: https://www.tensorflow.org/tutorials/wide_and_deep
    """

    def __init__(self,
                 number_embeddings_features, tokens_in_embeddings,
                 deep_parameters, deep_activation_fn, drop_out_rate=0.0):

        super(WideDeepClassifier, self).__init__(name="WideAndDeep")

        # Embeddings
        # Make the axes
        self.luts = []

        for e in range(len(number_embeddings_features)):
            init_uniform = UniformInit(0, 1)

            # pad_idx have to be initialize to 0 explicitly.

            lut = LookupTable(tokens_in_embeddings[e], number_embeddings_features[e],
                              init_uniform, pad_idx=0, update=True)

            self.luts.append(lut)

        # Model specification

        init_xavier = XavierInit()

        layers = []
        for i in range(len(deep_parameters)):
            layers.append(Affine(nout=deep_parameters[i], weight_init=init_xavier,
                                 activation=deep_activation_fn))
            if drop_out_rate > 0.0:
                layers.append(Dropout(keep=drop_out_rate))

        layers.append(Affine(axes=tuple(), weight_init=init_xavier))

        self.deep_layers = Sequential(layers)

        self.linear_layer = Affine(axes=tuple(), weight_init=init_xavier)

    @SubGraph.scope_op_creation
    def __call__(self, batch_size, placeholders):

        embedding_ops = []

        for idx, lut in enumerate(self.luts):
            embedding_op = lut(placeholders['embeddings_placeholders'][idx])

            embedding_ops.append(embedding_op)

        X_deep = ng.concat_along_axis([placeholders['X_d']] + embedding_ops,
                                      ng.make_axis(name="F"))

        self.wide_deep = ng.sigmoid(self.deep_layers(X_deep) +
                                    self.linear_layer(placeholders['X_w'])
                                    + ng.variable((), initial_value=0.5).named('b'))

        return self.wide_deep
