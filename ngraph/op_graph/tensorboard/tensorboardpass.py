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

from ngraph.transformers.passes.passes import GraphPass
from ngraph.op_graph.tensorboard.tensorboard import TensorBoard


class TensorBoardPass(GraphPass):
    """
    A pass that saves graph for TensorBoard graph dispaly

    Arguments:
        logdir: directory to save the log
    """
    def __init__(self, logdir):
        super(TensorBoardPass, self).__init__()
        self.logdir = logdir

    def do_pass(self, ops):
        tb = TensorBoard(self.logdir)
        tb.add_graph(ops)
        return ops
