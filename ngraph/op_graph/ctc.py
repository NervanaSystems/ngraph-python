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
from ngraph.op_graph.op_graph import TensorOp


def ctc(utts, lbls, utt_lens, lbl_lens, grads, out_axes):
    """
    Computes the CTC op
    Args:
        utts (TensorOp): Activations induced by the utterances.
        lbls (TensorOp): Transcripts corresponding to the utterances.
        utt_lens (TensorOp): (Strided) utterance lengths.
        lbl_lens (TensorOp): Transcript lengths.
        grads (TensorOp): Buffer holding the gradients.
        out_axes (Axes): The axes for the output.

    Returns:
        TensorOp: The result of the CTC op.

    """
    return CTCOp(utts, lbls, utt_lens, lbl_lens, grads, out_axes)


class CTCOp(TensorOp):

    def __init__(self, utts, lbls, utt_lens, lbl_lens, grads, out_axes, **kwargs):
        """
        Arguments:
            utts: Activations induced by the utterances.
            lbls: Transcripts corresponding to the utterances.
            utt_lens: (Strided) utterance lengths.
            lbl_lens: Transcript lengths.
            grads: CTC gradients.
            out_axes: The axes for the output.
        """

        # verify shapes
        if len(utts.shape) != 3:
            raise ValueError((
                'utts shape must have length 3, found {}'
            ).format(len(utts.shape)))

        if len(lbls.shape) != 1:
            raise ValueError((
                'lbls shape must have length 1, found {}'
            ).format(len(lbls.shape)))

        if len(utt_lens.shape) != 1:
            raise ValueError((
                'utt_lens shape must have length 1, found {}'
            ).format(len(utt_lens.shape)))

        if len(lbl_lens.shape) != 1:
            raise ValueError((
                'lbl_lens shape must have length 1, found {}'
            ).format(len(lbl_lens.shape)))

        if len(grads.shape) != 3:
            raise ValueError((
                'grads shape must have length 3, found {}'
            ).format(len(grads.shape)))

        self.grads = grads

        super(CTCOp, self).__init__(args=(utts, lbls, utt_lens, lbl_lens, grads), 
                                    axes=out_axes, 
                                    **kwargs)

    def generate_adjoints(self, adjoints, delta, utts, lbls, utt_lens, lbl_lens, grads):
        """
        TODO
        """
        utts.generate_add_delta(adjoints, self.grads)

