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

from ngraph.transformers.gpu.gpulayout import DimshuffleOp
from ngraph.transformers.passes.passes import GraphPass, PeepholeGraphPass
from ngraph.util.generics import generic_method
from ngraph.op_graph.op_graph import Op, tdcache
from ngraph.op_graph.pooling import PoolingOp, BpropPoolOp
from ngraph.flex import gpuflex16


class FlexDtypePass(PeepholeGraphPass):
    @generic_method(dispatch_base_type=Op)
    def visit(self, op, *args):
        # TODO currently hard coded gpuflex16
        op.dtype = gpuflex16


class FlexPropagateEntryPass(PeepholeGraphPass):

    def __init__(self, transformer, **kwargs):
        super(FlexPropagateEntryPass, self).__init__(**kwargs)
        self.transformer = transformer
        self.propagate_flex_entry = False

    @generic_method(dispatch_base_type=Op)
    def visit(self, op, *args):
        # copy flex entry for any op followed by Dimshuffle op, PoolingOp or BpropPoolOp
        if self.propagate_flex_entry:
            if isinstance(op, (DimshuffleOp, PoolingOp, BpropPoolOp)):
                self.transformer.get_op_tensor(op).flex_entry = self.flex_entry
                self.propagate_flex_entry = False
        if op.is_tensor_op:
            self.propagate_flex_entry = True
            self.flex_entry = self.transformer.get_op_tensor(op).flex_entry


class ClearTensorDescriptions(GraphPass):
    def __init__(self, transformer, **kwargs):
        self.transformer = transformer
        super(ClearTensorDescriptions, self).__init__(**kwargs)

    def do_pass(self, ops):
        self.transformer.initialize_allocations()
        tdcache.tensor_description_cache.clear()
        return ops
