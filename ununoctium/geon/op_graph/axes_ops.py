from geon.op_graph.op_graph import TensorOp


class dimshuffle(TensorOp):
    """
    NOTE:
    The numpy implementation returns a view whenever possible ... not sure yet
    how to incorporate that logic into here since we need to know before the
    function returns if we need to allocate space or not.
    """

    def __init__(self, x, axes, **kwargs):
        """
        Shuffle the axes of x so that they are in the order specified in axes.

        All Axis in x.axes must also be present in axes.
        """
        kwargs['axes'] = axes
        super(dimshuffle, self).__init__(args=(x,), **kwargs)

        # determine the new order of the axes (used by numpy)
        # TODO: move somewhere else, potentially axes, nptransformer, somewhere
        # else?
        self.axes_order = x.tensor_description().dimshuffle_positions(axes)

    def call_info(self):
        """
        Returns TensorDescription of input Op x.
        """
        return [self.args[0].tensor_description()]

    def generate_adjoints(self, adjoints, delta, input):
        """
        The derivative of dimshuffle is a dimshuffle in the opposite order.
        Dimshuffle the deltas back into same order as the input (x).
        """
        input.generate_add_delta(
            adjoints, dimshuffle(delta, input.axes)
        )
