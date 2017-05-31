import ngraph as ng
from ngraph.transformers.passes.passes import GraphRewritePass
from ngraph.op_graph.op_graph import PatternLabelOp


class FlexFusion(GraphRewritePass):

    def construct_sigmoid_pattern(self):
        """
        Generate graph op that represents a pattern for Sigmoid operation
        ng.sigmoid(x)

        Returns:
            Single pattern that matches Sigmoid

        """

        # We want to match x tensor for Sigmoid.
        self.sigmoid_x_label = "X"
        # We bind op to X unconditionally.
        x = PatternLabelOp(self.sigmoid_x_label, axes={ng.make_axis(name='N')})
        sigmoid_op = ng.sigmoid(x)
        return sigmoid_op

    def fuse_sigmoid_callback(self, op, label_map_op_list):
        """
        Callback function that handles fusion for sigmoid  pattern
        """
        for (label_map, op) in label_map_op_list:
            # Matched Sigmoid pattern, do the replacement here.
            x = label_map[self.sigmoid_x_label]
            sigmoid_atomic_op = ng.sigmoidAtomic(x)
            self.replace_op(op, sigmoid_atomic_op)

    def construct_ce_binnary_inner_pattern(self):
        """
        Generate graph op that represents a pattern for Cross Entropy Binnary Inner operation
        ng.cross_entropy_binary_inner(y, t, enable_sig_opt=True, enable_diff_opt=True)

        Returns:
            Single pattern that matches Sigmoid

        """

        # We want to match x tensor for Sigmoid.
        self.ce_x_label = "X"
        self.ce_y_label = "Y"
        self.ce_t_label = "T"

        x = PatternLabelOp(self.ce_x_label, axes={ng.make_axis(name='N')})
        y = PatternLabelOp(self.ce_y_label, axes={ng.make_axis(name='N')})
        t = PatternLabelOp(self.ce_t_label, axes={ng.make_axis(name='N')})

        y.deriv_handler = ng.SigmoidOp(x)

        cross_op = ng.cross_entropy_binary_inner(y, t, enable_sig_opt=True, enable_diff_opt=True)
        return cross_op

    def fuse_ce_binnary_inner_callback(self, op, label_map_op_list):
        """
        Callback function that handles fusion for cross_entropy_binnary_inner pattern
        """
        for (label_map, op) in label_map_op_list:
            # Matched Sigmoid pattern, do the replacement here.
            y = label_map[self.ce_y_label]
            t = label_map[self.ce_t_label]

            cross_without_opt_op = ng.cross_entropy_binary_inner(y, t, enable_sig_opt=False,
                                                                 enable_diff_opt=True)
            self.replace_op(op, cross_without_opt_op)

    def __init__(self):
        # Register Sigmoid pattern
        pattern_sigmoid = self.construct_sigmoid_pattern()
        self.register_pattern(pattern_sigmoid, self.fuse_sigmoid_callback)

        # Register cross_entropy_binnary_inner pattern
        pattern_ce_binnary_inner = self.construct_ce_binnary_inner_pattern()
        self.register_pattern(pattern_ce_binnary_inner, self.fuse_ce_binnary_inner_callback)
