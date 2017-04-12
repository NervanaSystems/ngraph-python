from ngraph.transformers.passes.passes import PeepholeGraphPass
from ngraph.util.generics import generic_method
from ngraph.op_graph.op_graph import Add, Maximum, Multiply, Minimum
from ngraph.op_graph.relu import ReluOp


class FusionPass(PeepholeGraphPass):

    @generic_method()
    def visit(self, op):
        """
        Base case.
        """
        pass

    def fuse_fprop_relu(self, input1, input2, op):
        """
        Max and min fused to relu for the IA transformer.
        """
        # expression is maximum(x,0) + slope*minimum(0,x)
        mul_arg1, mul_arg2 = input2.args
        max_arg1, max_arg2 = input1.args
        if(max_arg1.is_scalar and max_arg1.args[0].tensor.const == 0
           and not(max_arg2.is_scalar)):  # check max(0,x) or max(x,0)
            input_tensor = max_arg2
        elif(max_arg2.is_scalar and max_arg2.args[0].tensor.const == 0
             and not(max_arg1.is_scalar)):
            input_tensor = max_arg1
        else:
            return
        # check if slope* Minimum(0,x)  or Minimum(0,x)* slope
        if(mul_arg1.is_scalar and not(mul_arg2.is_scalar) and isinstance(mul_arg2, Minimum)):
            input_slope, = mul_arg1.args
            min_arg1, min_arg2 = mul_arg2.args
            if not ((min_arg1.is_scalar and min_arg1.args[0].tensor.const == 0
                    and not(min_arg2.is_scalar)) and min_arg2 == input_tensor):  # Min(0,x)/(x,0)
                if not ((min_arg2.is_scalar and min_arg2.args[0].tensor.const == 0
                        and not(min_arg1.is_scalar)) and min_arg1 == input_tensor):
                    return
            new_op = ReluOp(input_tensor, input_slope.tensor.const)
            self.replace_op(op, new_op)
        elif(mul_arg2.is_scalar and not(mul_arg1.is_scalar) and isinstance(mul_arg1, Minimum)):
            input_slope, = mul_arg2.args
            min_arg1, min_arg2 = mul_arg1.args
            if not ((min_arg1.is_scalar and min_arg1.args[0].tensor.const == 0
                    and not(min_arg2.is_scalar)) and min_arg2 == input_tensor):
                if not ((min_arg2.is_scalar and min_arg2.args[0].tensor.const == 0
                        and not(min_arg1.is_scalar)) and min_arg1 == input_tensor):
                    return
            new_op = ReluOp(input_tensor, input_slope.tensor.const)
            self.replace_op(op, new_op)
        else:
            return

    @visit.on_type(Add)
    def visit(self, op):
        """
        Max and min fused to relu for the IA transformer.
        """
        # expression is maximum(x,0) + slope*minimum(0,x)
        input1, input2 = op.args
        if (isinstance(input1, Maximum) and isinstance(input2, Multiply)):
            self.fuse_fprop_relu(input1, input2, op)
        elif (isinstance(input2, Maximum) and isinstance(input1, Multiply)):
            self.fuse_fprop_relu(input2, input1, op)
