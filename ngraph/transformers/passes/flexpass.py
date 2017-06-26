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
    def __init__(self):
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
    def do_pass(self, ops, transformer):
        transformer.initialize_allocations()
        tdcache.tensor_description_cache.clear()
        return ops, transformer
