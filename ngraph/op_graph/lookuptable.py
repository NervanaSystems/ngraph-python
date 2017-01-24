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
from ngraph.op_graph.op_graph import TensorOp, ContiguousOp


def lookuptable(lut, idx, axes, update=True, pad_idx=None, docstring=None):
    """
    An operation to do the lookup from lut using the idx.
    Output axes are given as well, so from lut and idx axes, it indicates which
    axis to do the lookup.

    Args:
        lut (TensorOp): The lookup table.
        idx (TensorOp): The indices to do the lookup.
        axes (Axes): output axes
        pad_idx (int): The int indicates the padding index
        docstring (String, optional): Documentation for the op.

    Returns:
        TensorOp: The result of the lookup.

    """
    return LookupTableOp(lut, idx, axes=axes, update=True, pad_idx=pad_idx, docstring=docstring)


class LookupTableOp(TensorOp):

    def __init__(self, lut, idx, axes, update=True, pad_idx=None, **kwargs):
        """
        Arguments:
            lut  : lookup tensor.
            idx : indices to lookup.

        Return:
        """

        # check shape of lut and index
        if len(lut.shape) != 2:
            raise ValueError((
                'lookup table shape must be length 2, found {}'
            ).format(len(lut.shape)))

        if len(idx.shape) != 1:
            raise ValueError((
                'index shape must be length 1, found {}'
            ).format(len(lut.shape)))

        # axes are the output axes, and it will indicate which axis to do the lookup
        # so one of the lut axis has to be in axes
        if not (lut.axes[0] in axes or lut.axes[1] in axes):
            raise ValueError((
                "Output axes must indicate which axis to do lookup.  "
                "None of the output axes is from lookup table."
                "Found lut axes: {lut_axes} "
                "Found output axes: {out_axes}."
            ).format(
                lut_axes=lut.axes,
                out_axes=axes,
            ))

        if not idx.axes[0] in axes:
            raise ValueError((
                "Output axes must index axes.  "
                "Found index axes: {idx_axes} "
                "Found output axes: {out_axes}."
            ).format(
                idx_axes=idx.axes,
                out_axes=axes,
            ))

        # decide the lookup axis based on the output axes and lut axes
        # if lut shape is (V, F), and output axes has F, the lut_axis is 0
        # lut_axis is the axis being indexed by idx
        self.lut_axis = 0 if lut.axes[1] in axes else 1
        self.pad_idx = pad_idx
        self.update = update

        if axes[self.lut_axis] != idx.axes[0]:
            raise ValueError("Cannot transpose lut axes implicitly")

        super(LookupTableOp, self).__init__(args=(lut, idx),
                                            axes=axes,
                                            **kwargs)

    def generate_adjoints(self, adjoints, delta, lut, idx):
        """
        TODO
        """
        lut.generate_add_delta(adjoints, update_lut(delta, lut, idx, self))


class LutDerivOp(TensorOp):
    """
    Maintains index and conv_params through forwarding of the original op.

    Arguments:
        fprop: The original op.
    """
    def __init__(self, fprop, **kwargs):
        super(LutDerivOp, self).__init__(**kwargs)
        self.fprop = fprop
        self.lut_axis = self.fprop.forwarded.lut_axis
        self.pad_idx = self.fprop.forwarded.pad_idx
        self.update = self.fprop.forwarded.update


class update_lut(LutDerivOp):
    def __init__(self, delta, lut, idx, fprop, **kwargs):
        """
        Arguments:
            lut  : lookup table.
            idx  : indices for lookup
        """
        super(update_lut, self).__init__(
            args=(ContiguousOp(delta), ContiguousOp(idx)),
            fprop=fprop,
            axes=lut.axes, **kwargs
        )


class bprop_lut(LutDerivOp):
    def __init__(self, delta, lut, idx, fprop, **kwargs):
        """
        Arguments:
            lut  : lookup table.
            idx  : indices for lookup
        """
        super(bprop_lut, self).__init__(
            args=(ContiguousOp(delta), ContiguousOp(lut)),
            fprop=fprop,
            axes=idx.axes, **kwargs
        )
