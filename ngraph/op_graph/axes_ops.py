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
from ngraph.op_graph.op_graph import TensorOp


class dimshuffle(TensorOp):
    """
    NOTE:
    The numpy implementation returns a view whenever possible ... not sure yet
    how to incorporate that logic into here since we need to know before the
    function returns if we need to allocate space or not.
    """

    def __init__(self, x, axes, **kwargs):
        """
        Shuffle the axes of x so that they are in the order specified in axes.

        All Axis in x.axes must also be present in axes.
        """
        kwargs['axes'] = axes
        super(dimshuffle, self).__init__(args=(x,), **kwargs)

        # determine the new order of the axes (used by numpy)
        # TODO: move somewhere else, potentially axes, nptransformer, somewhere
        # else?
        self.axes_order = x.tensor_description().dimshuffle_positions(axes)

    def call_info(self):
        """
        Returns TensorDescription of input Op x.
        """
        return [self.args[0].tensor_description()]

    def generate_adjoints(self, adjoints, delta, input):
        """
        The derivative of dimshuffle is a dimshuffle in the opposite order.
        Dimshuffle the deltas back into same order as the input (x).
        """
        input.generate_add_delta(
            adjoints, dimshuffle(delta, input.axes)
        )
