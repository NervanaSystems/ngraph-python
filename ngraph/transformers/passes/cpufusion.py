import ngraph as ng
from ngraph.transformers.passes.passes import GraphRewritePass, PeepholeGraphPass
from ngraph.util.generics import generic_method
from ngraph.transformers.cpu.relu import ReluOp, BpropReluOp
from ngraph.op_graph.op_graph import Add, Multiply, Greater, Less
from ngraph.op_graph.op_graph import Maximum, Minimum, BroadcastOp
from ngraph.op_graph.op_graph import ReciprocalOp, Subtract, SqrtOp, AssignableTensorOp, variable, TensorOp
from ngraph.op_graph.op_graph import PatternLabelOp, PatternSkipOp
from ngraph.op_graph.op_graph import Unflatten, ContiguousOp, BroadcastOp, BinaryElementWiseOp, Flatten, Divide
from ngraph.op_graph.batchnorm import BatchnormOp
from collections import deque as Queue


class CPUFusion(GraphRewritePass):

    def construct_relu_fprop_pattern(self):
        """
        Generate graph op that represents a pattern for Relu operation.
        max(val, 0) + slope * min (0, val)

        Note that there could be multiple patterns:
        Pattern 1 - max(x, 0) + slope * min (0, x)
        Pattern 2 - max(0, x) + slope * min (0, x)
        ..
        But we generate only 1 and match_pattern takes care of matching all
        permutations.

        Returns:
            Single pattern that matches Relu fprop op

        """
        zero = ng.constant(0)
        zero_w_broadcast = PatternSkipOp(zero,
                                         (lambda op:
                                          isinstance(op, BroadcastOp)))
        # We want to match x tensor and slope for Relu.
        self.relu_fwd_slope_label = "S"
        self.relu_fwd_x_label = "X"
        # We bind op to X unconditionally.
        x = PatternLabelOp(self.relu_fwd_x_label)
        max_op = Maximum(x, zero_w_broadcast)
        # We bind slope op to S only if it is scalar.
        slope_label_op = PatternLabelOp(self.relu_fwd_slope_label,
                                        (lambda op: op.is_scalar))
        slope = PatternSkipOp(slope_label_op,
                              (lambda op: isinstance(op, BroadcastOp)))
        min_op = Minimum(zero_w_broadcast, x)
        mul_op = Multiply(slope, min_op)
        add_op = Add(max_op, mul_op)
        return add_op

    def fuse_relu_fprop_callback(self, op, label_map_op_list):
        """
        Callback function that handles fusion for Relu fprop pattern
        """
        for (label_map, op) in label_map_op_list:
            # Matched Relu pattern, do the replacement here.
            x = label_map[self.relu_fwd_x_label]
            slope = label_map[self.relu_fwd_slope_label]
            relu_fwd_op = ReluOp(x, slope.tensor.const)
            # We need to store relu_fwd_op in a dictionary so that backward Relu
            # can access it.
            self.tensor_to_op_dict[x] = relu_fwd_op
        self.replace_op(op, relu_fwd_op)

    def construct_relu_bprop_pattern(self):
        """
        Generate graph op that represents a pattern for Relu backprop operation.
        delta * greater(x, 0) + delta * slope * less(x, 0)

        Returns:
            Single pattern that matches Relu bprop op

        """
        # We want to match x tensor, slope and delta for Relu.
        self.relu_bwd_slope_label = "S"
        self.relu_bwd_x_label = "X"
        self.relu_bwd_delta_label = "D"

        # construct 1st operand of Add
        zero = ng.constant(0)
        zero_w_broadcast = ng.PatternSkipOp(zero,
                                            (lambda op:
                                             isinstance(op, BroadcastOp)))
        x = ng.PatternLabelOp(self.relu_bwd_x_label,
                              (lambda op: not op.is_scalar))  # X is not scalar.
        greater_op = Greater(x, zero_w_broadcast)
        delta = PatternLabelOp(self.relu_bwd_delta_label,
                               (lambda op: not op.is_scalar))  # delta is not scalar.
        mul_greater_delta_op = Multiply(greater_op, delta)

        # Construct 2nd operand of Add
        # We bind slope op to S only if it is scalar.
        slope = PatternLabelOp(self.relu_bwd_slope_label,
                               (lambda op: op.is_scalar))
        less_op = Less(x, zero_w_broadcast)
        mul_slope_delta_op = Multiply(slope, delta)
        mul_slope_delta_less_op = Multiply(less_op, mul_slope_delta_op)

        add_op = Add(mul_greater_delta_op, mul_slope_delta_less_op)
        return add_op

    def fuse_relu_bprop_callback(self, op, label_map_op_list):
        """
        Callback function that handles fusion for Relu bprop pattern
        """
        for (label_map, op) in label_map_op_list:
            # Matched Relu pattern, do the replacement here.
            x = label_map[self.relu_bwd_x_label]
            delta = label_map[self.relu_bwd_delta_label]
            relu_fprop = self.tensor_to_op_dict[x]
            self.replace_op(op, BpropReluOp(delta, x, relu_fprop))

    def __init__(self):
        self.tensor_to_op_dict = dict()

        # Register Relu fprop pattern
        pattern_relu_fprop = self.construct_relu_fprop_pattern()
        self.register_pattern(pattern_relu_fprop, self.fuse_relu_fprop_callback)

        # Register Relu bprop pattern
        pattern_relu_bprop = self.construct_relu_bprop_pattern()
        self.register_pattern(pattern_relu_bprop, self.fuse_relu_bprop_callback)


# Delete after moving Batchnorm to CPUFusion
class FusionPass(PeepholeGraphPass):


    @generic_method()
    def visit(self, op):
        """
        Base case.
        """
        pass

    def fuse_fprop_batch_norm(self, dict_ops_to_params, op):
        inputs = dict_ops_to_params["in_obj"]
        gamma = dict_ops_to_params["gamma"]
        bias = dict_ops_to_params["bias"]
        variance = BroadcastOp(dict_ops_to_params["variance"], axes=inputs.axes)
        epsilon = dict_ops_to_params["epsilon"].args[0].tensor.const
        mean = BroadcastOp(dict_ops_to_params["mean"], axes=inputs.axes)
        new_op = BatchnormOp(inputs, gamma, bias, epsilon, mean, variance)
        self.replace_op(op, new_op)

    def check_for_pattern(self, args1, args2, op_type1, op_type2):
        """
        check if the subgraph matches the expected pattren across all the inputs
        """
        if (isinstance(args1, op_type1) and isinstance(args2, op_type2)) or \
                (isinstance(args1, op_type2) and isinstance(args2, op_type1)):
            return True
        else:
            return False

    def map_ops_to_batch_params(self, key, op, arg_list, op_dict):
        """
        For a given op, match the op.args to the corrosponding keywords in the arg_list
        ex: if op = Add, op_dict["in_obj"] = FalttenOp; op_dict["mean"]=Divide
        """
        if len(arg_list) > 1:
            if isinstance(op, Subtract):
                if (isinstance(arg_list[0], Flatten)):
                    op_dict[key[0]] = arg_list[0]
                    op_dict[key[1]] = arg_list[1]
                else:
                    op_dict[key[1]] = arg_list[0]
                    op_dict[key[0]] = arg_list[1]

            elif isinstance(op, Add):
                if (isinstance(arg_list[0], BinaryElementWiseOp)):
                    op_dict[key[0]] = arg_list[1]
                    op_dict[key[1]] = arg_list[0]
                else:
                    op_dict[key[0]] = arg_list[0]
                    op_dict[key[1]] = arg_list[1]

        else:
            op_dict[key[0]] = arg_list[0]

    @visit.on_type(Add)
    def visit(self, op):
        """
        self.gamma * ((in_obj - xmean) * ng.reciprocal(ng.sqrt(xvar + self.eps))) + self.beta
        is fused to BatchNorm Op for IA transformer
        """
        # Op dictionary which maps between op and op type
        keys = ["mean", "variance", "gamma", "bias", "in_obj", "epsilon"]
        op_dict = {key: None for key in keys}
        root_op = op

        # queue to maintain the list of op's, op will be remove once it is visited
        next_op = Queue()
        next_op.append(op)

        # keep track's of tree level
        level = 0

        # BFS to find if the sub graph matches the expected pattren
        while next_op:
            op = next_op.popleft()
            op_args = op.args

            if (isinstance(op, Add) and (level == 0)):
                if (self.check_for_pattern(op_args[0], op_args[1], Multiply, BroadcastOp)):
                    if isinstance(op_args[0], BinaryElementWiseOp):
                        self.map_ops_to_batch_params(["bias"], op, [op_args[1]], op_dict)
                        next_op.append(op_args[0])
                    elif isinstance(op_args[1], BinaryElementWiseOp):
                        self.map_ops_to_batch_params(["bias"], op, [op_args[0]], op_dict)
                        next_op.append(op_args[1])
                    level = level + 1

            elif (isinstance(op, Multiply) and (level == 1)):
                if (self.check_for_pattern(op_args[0], op_args[1], Multiply, BroadcastOp)):
                    if isinstance(op_args[0], BinaryElementWiseOp):
                        self.map_ops_to_batch_params(["gamma"], op, [op_args[1]], op_dict)
                        next_op.append(op_args[0])
                    elif isinstance(op_args[1], BinaryElementWiseOp):
                        self.map_ops_to_batch_params(["gamma"], op, [op_args[0]], op_dict)
                        next_op.append(op_args[1])
                    level = level + 1

            elif ((level == 2) and isinstance(op, Multiply)):
                if (self.check_for_pattern(op_args[0], op_args[1], Subtract, BroadcastOp)):
                    next_op.append(op_args[0])
                    next_op.append(op_args[1])
                    level = level + 1

            elif ((level == 3) and isinstance(op, Subtract)):
                self.map_ops_to_batch_params(["in_obj", "mean"], op, [op_args[0], op_args[1]], op_dict)

            elif ((level == 3) and isinstance(op, BroadcastOp)):
                next_op.append(op_args[0])
                level = level + 1

            elif ((level == 4) and isinstance(op, ReciprocalOp)):
                next_op.append(op_args[0])
                level = level + 1

            elif ((level == 5) and isinstance(op, SqrtOp)):
                next_op.append(op_args[0])
                level = level + 1

            elif ((level == 6) and isinstance(op, Add)):
                if (self.check_for_pattern(op_args[0], op_args[1], Divide, BroadcastOp)):
                    self.map_ops_to_batch_params(["epsilon", "variance"], op, [op_args[0], op_args[1]], op_dict)
                    level = level + 1

        # if we reach the correct depth and all the pattern matches then fuse to form BatchnormOp
        if (level == 7):
            self.fuse_fprop_batch_norm(op_dict, root_op)
