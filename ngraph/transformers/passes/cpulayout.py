from ngraph.transformers.passes.passes import PeepholeGraphPass
from ngraph.util.generics import generic_method
from ngraph.op_graph.op_graph import Op, ContiguousOp, Add
from ngraph.op_graph.convolution import ConvolutionOp, update_conv
from ngraph.op_graph.pooling import PoolingOp
from ngraph.op_graph.ctc import CTCOp


def is_contiguous(op):
    return isinstance(op, ContiguousOp)


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
        # if not isinstance(inputs, ContiguousOp):
        #    inputs = ContiguousOp(inputs)
        #    replace = True

        if not isinstance(filters, ContiguousOp):
            filters = ContiguousOp(filters)
            replace = True

        if replace:
            self.replace_op(op, ConvolutionOp(op.conv_params, inputs, filters, axes=op.axes))

    @visit.on_type(update_conv)
    def visit(self, op):
        delta = op.args[0]
        inputs = op.args[1]

        replace = False
        if not isinstance(delta, ContiguousOp):
            delta = ContiguousOp(delta)
            replace = True

        # if not isinstance(inputs, ContiguousOp):
        #    inputs = ContiguousOp(inputs)
        #    replace = True

        if replace:
            self.replace_op(op, update_conv(delta, inputs, op.fprop.args[1], op.fprop))

    @visit.on_type(CTCOp)
    def visit(self, op):
        """
        Warp-CTC requires all args to be contiguous
        """
        replace = False
        args = list(op.args)
        for ii, arg in enumerate(args):
            if not is_contiguous(arg):
                args[ii] = ContiguousOp(arg)
                replace = True

        if replace is True:
            self.replace_op(op, CTCOp(*args, axes=op.axes))

    @visit.on_type(Add)
    def visit(self, op):

        input1, input2 = op.args
        replace = False

        if not isinstance(input1, ContiguousOp):
            input1 = ContiguousOp(input1)
            replace = True

        if not isinstance(input2, ContiguousOp):
            input2 = ContiguousOp(input2)
            replace = True

        if replace:
            self.replace_op(op, Add(input1, input2))

    @visit.on_type(PoolingOp)
    def visit(self, op):
        """
        MKLDNN Pooling implementation requires contiguous layout.
        """
        inputs = op.args[0]
        if not isinstance(inputs, ContiguousOp):
            new_op = PoolingOp(op.pool_params, ContiguousOp(inputs), axes=op.axes)
            self.replace_op(op, new_op)
