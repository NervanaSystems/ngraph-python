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


class PrintOp(TensorOp):
    """
    Prints the value of a tensor at every evaluation of the Op.  Has a nop
    adjoint.

    This is easy right now in CPU transformers, but will be more annoying to
    implement for other devices.

    I imagine there will be a much better way to do this in the future.  For
    now, it is a handy hack.
    """

    def __init__(self, x, prefix=None, **kwargs):
        """
        Arguments:
            x: the Op to print at each graph execution
            prefix: will be cast as a string and printed before x as a prefix
        """
        if prefix is not None:
            prefix = str(prefix)

        self.prefix = str(prefix)

        kwargs['axes'] = x.axes
        super(PrintOp, self).__init__(args=(x,), **kwargs)

    def generate_adjoints(self, adjoints, delta, x):
        """
        adjoints pass through PrintOp unchanged.
        """
        x.generate_add_delta(adjoints, delta)
