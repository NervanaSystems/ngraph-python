# ----------------------------------------------------------------------------
# Copyright 2017 Nervana Systems Inc.
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
from ngraph.op_graph.op_graph import TensorOp, persistent_tensor, sequential


def ctc(activations, labels, activation_lens, label_lens, axes=None):
    """
    Computes the CTC cost using warp-ctc
    Args:
        activations (TensorOp): The network output to compare against the transcripts
        labels (TensorOp): One-hot encoded transcript labels
        activation_lens (TensorOp): Length of activations for each example in the batch
        label_lens (TensorOp): Transcript length for each example in the batch
        axes (Axes, optional): Output axes for the cost tensor. Defaults to batch axis.

    Returns:
        TensorOp: The result of the CTC op.

    References:
        Graves, A., et al. (2006). https://doi.org/10.1145/1143844.1143891
        warp-ctc: https://github.com/baidu-research/warp-ctc

    """
    grads = persistent_tensor(axes=activations.axes).named("ctc_gradients")

    return CTCOp(activations, labels, activation_lens, label_lens, grads, axes=axes)


class CTCOp(TensorOp):

    def __init__(self, activations, labels, activation_lens, label_lens, grads,
                 axes=None, **kwargs):
        """
        Args:
            activations (TensorOp): The network output to compare against the transcripts
            labels (TensorOp): One-hot encoded transcript labels
            activation_lens (TensorOp): Length of activations for each example in the batch
            label_lens (TensorOp): Transcript length for each example in the batch
            grads (TensorOp): A persistent tensor that will be filled with CTC gradients
            axes (Axes, optional): Output axes for the cost tensor. Defaults to batch axis.
        """

        # verify shapes
        if len(activations.shape) != 3:
            raise ValueError(('activations must have 3 dimensions, ',
                              'found {}').format(len(activations.shape)))

        if activations.axes.batch_axis() is None:
            raise ValueError('activations must have a batch axis')

        if activations.axes.recurrent_axis() is None:
            raise ValueError('activations must have a recurrent axis')

        if len(labels.shape) != 1:
            raise ValueError(('labels 1must have 1 dimension, ',
                              'found {}').format(len(labels.shape)))

        if len(activation_lens.shape) != 1:
            raise ValueError(('activation_lens must have 1 dimension, ',
                              'found {}').format(len(activation_lens.shape)))

        if len(label_lens.shape) != 1:
            raise ValueError(('label_lens must have 1 dimension, ',
                              'found {}').format(len(label_lens.shape)))

        if axes is None:
            axes = activations.axes.batch_axes()

        super(CTCOp, self).__init__(args=(activations, labels,
                                          activation_lens, label_lens,
                                          grads),
                                    axes=axes,
                                    **kwargs)

    def generate_adjoints(self, adjoints, delta,
                          activations, labels, activation_lens, label_lens, grads):
        """
        Add gradients computed by warp-ctc do adjoints
        """
        activations.generate_add_delta(adjoints, sequential([self, grads]))
