from ngraph.transformers.passes.passes import PeepholeGraphPass
from ngraph.util.generics import generic_method
from ngraph.op_graph.op_graph import Op, ContiguousOp
from ngraph.op_graph.convolution import ConvolutionOp


class CPUTensorLayout(PeepholeGraphPass):
    """TODO."""
    @generic_method(dispatch_base_type=Op)
    def visit(self, op):
        """
        Base case.
        """
        pass

    @visit.on_type(ConvolutionOp)
    def visit(self, op):
        """
        Convolution implementation requires contiguous layout.
        """
        inputs, filters = op.args

        replace = False
        if not isinstance(inputs, ContiguousOp):
            inputs = ContiguousOp(inputs)
            replace = True

        if not isinstance(filters, ContiguousOp):
            filters = ContiguousOp(filters)
            replace = True

        if replace:
            self.replace_op(op, ConvolutionOp(op.conv_params, inputs, filters, axes=op.axes))
