from ngraph.transformers.gpu.gpulayout import DimshuffleOp
from ngraph.transformers.passes.passes import GraphPass, PeepholeGraphPass
from ngraph.util.generics import generic_method
from ngraph.op_graph.op_graph import Op, tdcache
from ngraph.flex import gpuflex16


class FlexDtypePass(PeepholeGraphPass):
    @generic_method(dispatch_base_type=Op)
    def visit(self, op, *args):
        # TODO currently hard coded gpuflex16
        op.dtype = gpuflex16


class FlexDECPass(PeepholeGraphPass):

    def __init__(self, **kwargs):
        super(FlexDECPass, self).__init__(**kwargs)
        self.propagate_flex_entry = False

    def do_pass(self, min_ops, transformer):
        self.transformer = transformer
        super(FlexDECPass, self).do_pass(min_ops, transformer)

    @generic_method(dispatch_base_type=Op)
    def visit(self, op, *args):
        # copy flex entry for any op followed by dimshuffle op
        if self.propagate_flex_entry:
            if isinstance(op, DimshuffleOp):
                self.transformer.get_op_tensor(op).flex_entry = self.flex_entry
                self.propagate_flex_entry = False
        if op.tensor_description():
            self.propagate_flex_entry = True
            self.flex_entry = self.transformer.get_op_tensor(op).flex_entry


class ClearTensorDescriptions(GraphPass):
    def do_pass(self, ops, transformer):
        transformer.initialize_allocations()
        tdcache.tensor_description_cache.clear()
        return ops, transformer
