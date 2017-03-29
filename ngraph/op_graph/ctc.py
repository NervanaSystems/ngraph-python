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
from ngraph.op_graph.op_graph import TensorOp, persistent_tensor


def ctc(activations, labels, activation_lens, label_lens, axes=None):
    """
    Computes the CTC op
    Args:
        activations (TensorOp): Activations induced by the utterances.
        labels (TensorOp): Transcripts corresponding to the utterances.
        activation_lens (TensorOp): (Strided) utterance lengths.
        label_lens (TensorOp): Transcript lengths.

    Returns:
        TensorOp: The result of the CTC op.

    """
    return CTCOp(activations, labels, activation_lens, label_lens, axes=axes)


class CTCOp(TensorOp):

    def __init__(self, activations, labels, activation_lens, label_lens, axes=None, **kwargs):
        """
        Arguments:
            activations: Activations induced by the utterances.
            labels: Transcripts corresponding to the utterances.
            activation_lens: (Strided) utterance lengths.
            label_lens: Transcript lengths.
        """

        # verify shapes
        if len(activations.shape) != 3:
            raise ValueError(('inputs must have 3 dimensions, ',
                              'found {}').format(len(activations.shape)))

        if len(labels.shape) != 1:
            raise ValueError(('labels must have 1 dimension, ',
                              'found {}').format(len(labels.shape)))

        if len(activation_lens.shape) != 1:
            raise ValueError(('input_lens must have 1 dimension, ',
                              'found {}').format(len(activation_lens.shape)))

        if len(label_lens.shape) != 1:
            raise ValueError(('label_lens must have 1 dimension, ',
                              'found {}').format(len(label_lens.shape)))

        super(CTCOp, self).__init__(args=(activations, labels, activation_lens, label_lens),
                                    axes=axes if axes is not None else activations.batch_axes(),
                                    **kwargs)
        self.grads = persistent_tensor(axes=activations.axes).named("ctc_gradients")

    def generate_adjoints(self, adjoints, delta, activations, labels, activation_lens, label_lens):
        """
        Add gradients computed by warp-ctc do adjoints
        """
        activations.generate_add_delta(adjoints, self.grads)

