from ngraph.transformers.passes.passes import PeepholeGraphPass
from ngraph.util.generics import generic_method
from ngraph.op_graph.op_graph import Add, Maximum, Multiply, Minimum, Greater, Less
from ngraph.transformers.cpu.relu import ReluOp, BpropReluOp


class FusionPass(PeepholeGraphPass):

    @generic_method()
    def visit(self, op):
        """
        Base case.
        """
        pass

    def __init__(self):
        self.dict = {}

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
            self.dict[input_tensor] = new_op
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
            self.dict[input_tensor] = new_op
        else:
            return

    def pattern_check_and_fuse_bprop_relu(self, mul_arg3, mul_arg4, input_tensor, delta, op):
        #  delta * slope * less(x, 0)
        less_arg1, less_arg2 = mul_arg3.args
        mul_arg5, mul_arg6 = mul_arg4.args
        if not ((less_arg1.is_scalar and less_arg1.args[0].tensor.const == 0
                and not(less_arg2.is_scalar) and less_arg2 == input_tensor)):
            if not ((less_arg2.is_scalar and less_arg2.args[0].tensor.const == 0 and
                    not(less_arg1.is_scalar) and less_arg1 == input_tensor)):
                return
        if ((mul_arg5.is_scalar and mul_arg5.args[0].tensor.const == 0
           and not(mul_arg6.is_scalar))):
            input_slope, = mul_arg5.args
        elif ((mul_arg6.is_scalar and mul_arg6.args[0].tensor.const == 0
              and not(mul_arg5.is_scalar))):
            input_slope, = mul_arg6.args
        else:
            return
        fprop = self.dict[input_tensor]
        new_op = BpropReluOp(delta, input_tensor, fprop)
        self.replace_op(op, new_op)

    def pattern_check_bprop_relu(self, mul_arg1, mul_arg2, mul_arg3, mul_arg4, op):
        # check Greater(x,0)* delta or delta*Greater(x,0)
        greater_arg1, greater_arg2 = mul_arg1.args
        delta = mul_arg2
        if ((greater_arg1.is_scalar and greater_arg1.args[0].tensor.const == 0
           and not(greater_arg2.is_scalar))):
            input_tensor = greater_arg2
        elif ((greater_arg2.is_scalar and greater_arg2.args[0].tensor.const == 0
               and not(greater_arg1.is_scalar))):
            input_tensor = greater_arg1
        else:
            return
        if(isinstance(mul_arg3, Less) and isinstance(mul_arg4, Multiply)):
            self.pattern_check_and_fuse_bprop_relu(mul_arg3, mul_arg4, input_tensor, delta, op)
        elif(isinstance(mul_arg4, Less) and isinstance(mul_arg3, Multiply)):
            self.pattern_check_and_fuse_bprop_relu(mul_arg4, mul_arg3, input_tensor, delta, op)

    def check_arg_ordering_bprop_relu(self, input1, input2, op):
        """
        Max and min fused to relu for the IA transformer.
        """
        mul_arg1, mul_arg2 = input1.args
        mul_arg3, mul_arg4 = input2.args
        if (isinstance(mul_arg1, Greater) and not(mul_arg2.is_scalar)):
            self.pattern_check_bprop_relu(mul_arg1, mul_arg2, mul_arg3, mul_arg4, op)
        elif (isinstance(mul_arg2, Greater) and not(mul_arg1.is_scalar)):
            self.pattern_check_bprop_relu(mul_arg2, mul_arg1, mul_arg3, mul_arg4, op)

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
        # expression is delta*greater(x, 0) + delta*slope*less(x, 0)
        elif (isinstance(input1, Multiply) and isinstance(input2, Multiply)):
            mul_arg1, mul_arg2 = input1.args
            mul_arg3, mul_arg4 = input2.args
            if (isinstance(mul_arg2, Greater) or isinstance(mul_arg1, Greater)):
                if (isinstance(mul_arg3, Multiply) and isinstance(mul_arg4, Less) or
                   isinstance(mul_arg4, Multiply) and isinstance(mul_arg3, Less)):
                    self.check_arg_ordering_bprop_relu(input1, input2, op)
            elif (isinstance(mul_arg1, Multiply) and isinstance(mul_arg2, Less) or
                  isinstance(mul_arg2, Multiply) and isinstance(mul_arg1, Less)):
                if (isinstance(mul_arg4, Greater) or isinstance(mul_arg3, Greater)):
                    self.check_arg_ordering_bprop_relu(input2, input1, op)
