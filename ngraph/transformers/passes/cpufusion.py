import ngraph as ng
from ngraph.op_graph.op_graph import Add, Multiply, Greater, Less
from ngraph.op_graph.op_graph import Maximum, Minimum, NegativeOp, Sum
from ngraph.op_graph.op_graph import ReciprocalOp, Subtract, SqrtOp
from ngraph.op_graph.op_graph import PatternLabelOp, PatternSkipOp
from ngraph.op_graph.op_graph import BroadcastOp, Flatten, Divide
from ngraph.op_graph.op_graph import DotOp, MapRolesOp, TensorValueOp, ContiguousOp
from ngraph.transformers.cpu.batchnorm import BatchnormOp, BpropBatchnormOp
from ngraph.transformers.cpu.relu import ReluOp, BpropReluOp
from ngraph.transformers.passes.passes import GraphRewritePass


class CPUFusion(GraphRewritePass):

    def construct_innerproduct_and_bias_pattern(self):
        """
        Pattern - Add(DotLowDimension, Bias).
        Returns:
            Single pattern that matches Add(DotLowDimension, Bias) pattern.
        """

        self.dot_x_label = "X"
        self.dot_y_label = "Y"
        self.bias_label = "B"
        x = PatternLabelOp(self.dot_x_label)
        y = PatternLabelOp(self.dot_y_label)

        bias_label_op = PatternLabelOp(self.bias_label,
                                       (lambda op: op.is_scalar) and
                                       (lambda op: isinstance(op, TensorValueOp)))
        bias = PatternSkipOp(bias_label_op,
                             (lambda op: isinstance(op, BroadcastOp)) or
                             (lambda op: isinstance(op, ContiguousOp)))

        dot_op = PatternSkipOp(DotOp(x, y, None),
                               (lambda op: isinstance(op, MapRolesOp)))
        add_op = Add(dot_op, bias)
        return add_op

    def fuse_innerproduct_and_bias_callback(self, op, label_map_op_list):
        """
        Callback function that handles fusion for Innerproduct + bias  pattern
        """
        for (label_map, op) in label_map_op_list:
            x = label_map[self.dot_x_label]
            y = label_map[self.dot_y_label]
            bias = label_map[self.bias_label]
            dot_new_op = DotOp(x, y, bias)
            self.replace_op(op, dot_new_op)

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

    def fuse_batchnorm_bprop_callback(self, op, label_map_op_list):
        """
        Callback function that handles fusion for batchnorm bprop pattern
        """
        for (label_map, op) in label_map_op_list:
            # Matched bprop batchnorm pattern, do the replacement here.
            input_tensor = label_map[self.batchnorm_bprop_input_tensor]
            delta = label_map[self.batchnorm_bprop_delta]
            if input_tensor in self.tensor_to_op_dict:
                batchnorm_fprop = self.tensor_to_op_dict[input_tensor]
                self.replace_op(op, BpropBatchnormOp(delta, input_tensor, batchnorm_fprop))
            else:
                assert("No matching fprop BatchnormOp for the input_tensor \
                       {}".format(input_tensor))

    def construct_batchnorm_bprop_pattern(self):
        """
        Generate graph op that represents a pattern for batchnorm backprop operation.
            dgamma = np.sum(delta * xhat)
            dbeta = np.sum(delta)
            dx = gamma_scale * (delta - (xhat * dgamma + dbeta) / m)
            In this pattern we are only generating the pattern for  dx.
        Returns:
               Single pattern that matches batchnorm bprop op
        """
        self.batchnorm_bprop_input_tensor = "input_tensor"
        self.batchnorm_bprop_delta = "delta"
        self.batchnorm_bprop_gamma_label = "gamma"
        self.batchnorm_bprop_var_label = "var"
        self.batchnorm_bprop_ivar_label = "ivar"
        self.batchnorm_bprop_xmu1_label = "xmu1"
        self.batchnorm_bprop_xmu2_label = "xmu2"
        self.batchnorm_bprop_negative_inverse_sqrtvar = "negative_inverse_sqrtvar"
        self.batchnorm_bprop_inverse_sqrtvar = "inverse_sqrtvar"
        self.batchnorm_bprop_sqrtvar_label = "sqrtvar"
        self.batchnorm_bprop_sqrsum = "sqrsum"
        self.batchnorm_bprop_mean_1 = "mean_1"
        self.batchnorm_bprop_mean_2 = "mean_2"
        self.batchnorm_bprop_input_sum = "input_sum"

        # bind the op's to the label
        input_tensor = PatternLabelOp(self.batchnorm_bprop_input_tensor,
                                      (lambda op: isinstance(op, Flatten)))
        var = PatternLabelOp(self.batchnorm_bprop_var_label,
                             (lambda op: isinstance(op, Divide)))
        gamma = PatternLabelOp(self.batchnorm_bprop_gamma_label,
                               (lambda op: isinstance(op, BroadcastOp)))
        delta = PatternLabelOp(self.batchnorm_bprop_delta,
                               (lambda op: isinstance(op, Flatten)))
        xmu1 = PatternLabelOp(self.batchnorm_bprop_xmu1_label,
                              (lambda op: isinstance(op, Subtract)))
        xmu2 = PatternLabelOp(self.batchnorm_bprop_xmu2_label,
                              (lambda op: isinstance(op, Subtract)))
        ivar = PatternLabelOp(self.batchnorm_bprop_ivar_label,
                              (lambda op: isinstance(op, BroadcastOp)))
        negative_inverse_sqrtvar = PatternLabelOp(self.batchnorm_bprop_negative_inverse_sqrtvar,
                                                  (lambda op: isinstance(op, NegativeOp)))
        inverse_sqrtvar = PatternLabelOp(self.batchnorm_bprop_inverse_sqrtvar,
                                         (lambda op: isinstance(op, ReciprocalOp)))
        sqrtvar = PatternLabelOp(self.batchnorm_bprop_sqrtvar_label,
                                 (lambda op: isinstance(op, SqrtOp)))
        sqrsum = PatternLabelOp(self.batchnorm_bprop_sqrsum,
                                (lambda op: isinstance(op, Sum)))
        mean_1 = PatternLabelOp(self.batchnorm_bprop_mean_1,
                                (lambda op: isinstance(op, Divide)))
        mean_2 = PatternLabelOp(self.batchnorm_bprop_mean_2,
                                (lambda op: isinstance(op, Divide)))
        input_sum = PatternLabelOp(self.batchnorm_bprop_input_sum,
                                   (lambda op: isinstance(op, Sum)))

        constant_point_5 = ng.constant(0.5)
        constant_point_5_w_broadcast = ng.PatternSkipOp(constant_point_5,
                                                        lambda op: isinstance(op, BroadcastOp))
        constant_two = ng.constant(2)
        constant_two_w_broadcast = ng.PatternSkipOp(constant_two,
                                                    lambda op: isinstance(op, BroadcastOp))
        # construct the pattern
        dxhat = Multiply(gamma, delta)
        # divar = np.sum(dxhat*xmu, axis=0)
        divar = Sum(Multiply(dxhat, xmu1))
        # dxmu1 = dxhat * ivar
        dxmu1 = Multiply(dxhat, ivar)
        # dsqrtvar = -1. /(sqrtvar**2) * divar
        dsqrtvar = Multiply(Multiply(inverse_sqrtvar, negative_inverse_sqrtvar), divar)
        # dvar = 0.5 * 1. /np.sqrt(var+eps) * dsqrtvar
        dvar = Divide(Multiply(dsqrtvar, constant_point_5_w_broadcast), sqrtvar)
        # dsq = 1. / N * np.ones((N, D)) * dvar
        dsq = Divide(Multiply(dvar, var), sqrsum)
        dsq_w_broadcast = ng.PatternSkipOp(dsq,
                                           (lambda op: isinstance(op, BroadcastOp)))
        # dxmu2 = 2 * xmu * dsq
        dxmu2 = Multiply(xmu2, Multiply(constant_two_w_broadcast, dsq_w_broadcast))

        # dx1 = (dxmu1 + dxmu2)
        # dmu = -1 * np.sum(dxmu1 + dxmu2, axis=0)
        # dx2 = 1. /N * np.ones((N,D)) * dmu
        # dx = dx1 + dx2
        dxmu2_mul = Multiply(Sum(ng.negative(dxmu2)), mean_2)
        dxmu2_div = Divide(dxmu2_mul, input_sum)
        dxmu2_div_w_broadcast = ng.PatternSkipOp(dxmu2_div,
                                                 (lambda op: isinstance(op, BroadcastOp)))
        dxmu2_div_plus_dxmu2 = Add(dxmu2_div_w_broadcast, dxmu2)

        dx1 = Add(dxmu1, dxmu2_div_plus_dxmu2)
        dxmu1_mul = Multiply(Sum(ng.negative(dxmu1)), mean_1)
        dxmu1_div = Divide(dxmu1_mul, Sum(input_tensor))
        dxmu1_div_w_broadcast = ng.PatternSkipOp(dxmu1_div,
                                                 (lambda op: isinstance(op, BroadcastOp)))
        dx = Add(dxmu1_div_w_broadcast, dx1)
        return dx

    def construct_batchnorm_fprop_pattern(self):
        """
        Generate graph op that represents a pattern for batchnorm fprop operation.
        self.gamma * ((in_obj - xmean) * ng.reciprocal(ng.sqrt(xvar + self.eps))) + self.beta
        Returns:
               Single pattern that matches batchnorm fprop op
        """
        self.batchnorm_fprop_input_tensor_label = "in_obj"
        self.batchnorm_fprop_gamma_label = "gamma"
        self.batchnorm_fprop_beta_label = "beta"
        self.batchnorm_fprop_variance_label = "variance"
        self.batchnorm_fprop_epsilon_label = "epsilon"
        self.batchnorm_fprop_mean_label = "mean"

        # bind the label to the op's which needed to be updated in the dict
        in_obj = PatternLabelOp(self.batchnorm_fprop_input_tensor_label,
                                (lambda op: isinstance(op, Flatten)))
        gamma = PatternLabelOp(self.batchnorm_fprop_gamma_label,
                               (lambda op: isinstance(op, BroadcastOp)))
        beta = PatternLabelOp(self.batchnorm_fprop_beta_label,
                              (lambda op: isinstance(op, BroadcastOp)))
        variance = PatternLabelOp(self.batchnorm_fprop_variance_label,
                                  (lambda op: isinstance(op, Divide)))
        epsilon = PatternLabelOp(self.batchnorm_fprop_epsilon_label,
                                 (lambda op: isinstance(op, BroadcastOp)))
        mean = PatternLabelOp(self.batchnorm_fprop_mean_label,
                              (lambda op: isinstance(op, BroadcastOp)))

        # construct the fprop batchnorm pattern matching the computation graph
        # ng.sqrt(xvar + self.eps)
        SqrtofVarianceAndEps = ng.sqrt(ng.add(variance, epsilon))
        # ng.reciprocal(ng.sqrt(xvar + self.eps))
        reciprocal_op = ng.reciprocal(SqrtofVarianceAndEps)
        reciprocal_op_w_braodcast = ng.PatternSkipOp(reciprocal_op,
                                                     lambda op: isinstance(op, BroadcastOp))
        # (in_obj - xmean) * ng.reciprocal(ng.sqrt(xvar + self.eps))
        mul_op_1 = ng.multiply(ng.subtract(in_obj, mean), reciprocal_op_w_braodcast)
        # "self.gamma * ((in_obj - xmean) * ng.reciprocal(ng.sqrt(xvar + self.eps)))
        MultiplyGamma = ng.multiply(mul_op_1, gamma)
        # self.gamma * ((in_obj - xmean) * ng.reciprocal(ng.sqrt(xvar + self.eps))) + self.beta
        AddBeta = ng.Add(MultiplyGamma, beta)
        return AddBeta

    def fuse_batchnorm_fprop_callback(self, op, label_map_op_list):
        """
        Callback function that handles fusion for batchnorm fprop pattern
        """
        for (label_map, op) in label_map_op_list:
            # Matched bprop batchnorm pattern, do the replacement here.
            inputs = label_map[self.batchnorm_fprop_input_tensor_label]
            gamma = label_map[self.batchnorm_fprop_gamma_label]
            beta = label_map[self.batchnorm_fprop_beta_label]
            variance = label_map[self.batchnorm_fprop_variance_label]
            mean = label_map[self.batchnorm_fprop_mean_label]
            epsilon = self.op_arg(label_map[self.batchnorm_fprop_epsilon_label],0).tensor.const
            batchnorm_fwd_op = BatchnormOp(inputs, gamma, beta, epsilon, mean, variance)

            # book keep the fprop batchnorm op to use during back propogation
            self.tensor_to_op_dict[inputs] = batchnorm_fwd_op
            self.replace_op(op, batchnorm_fwd_op)

    def __init__(self, **kwargs):
        super(CPUFusion, self).__init__(**kwargs)
        self.tensor_to_op_dict = dict()

        # Register Relu fprop pattern
        pattern_relu_fprop = self.construct_relu_fprop_pattern()
        self.register_pattern(pattern_relu_fprop, self.fuse_relu_fprop_callback)

        # Register Relu bprop pattern
        pattern_relu_bprop = self.construct_relu_bprop_pattern()
        self.register_pattern(pattern_relu_bprop, self.fuse_relu_bprop_callback)

        # Register batchnorm fprop pattern
        pattern_batchnorm_fprop = self.construct_batchnorm_fprop_pattern()
        self.register_pattern(pattern_batchnorm_fprop, self.fuse_batchnorm_fprop_callback)

        # Register Batchnorm bprop pattern
        pattern_batchnorm_bprop = self.construct_batchnorm_bprop_pattern()
        self.register_pattern(pattern_batchnorm_bprop, self.fuse_batchnorm_bprop_callback)

        # Register Inner + Bias  pattern
        pattern_inner_bias = self.construct_innerproduct_and_bias_pattern()
        self.register_pattern(pattern_inner_bias, self.fuse_innerproduct_and_bias_callback)
