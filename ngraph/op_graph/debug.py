from ngraph.op_graph.op_graph import TensorOp


class PrintOp(TensorOp):
    """
    Prints the value of a tensor at every evaluation of the Op.  Has a nop
    adjoint.

    This is easy right now in CPU transformers, but will be more annoying to
    implement for other devices.

    I imagine there will be a much better way to do this in the future.  For
    now, it is a handy hack.
    """

    def __init__(self, x, prefix=None, **kwargs):
        """
        Arguments:
            x: the Op to print at each graph execution
            prefix: will be cast as a string and printed before x as a prefix
        """
        if prefix is not None:
            prefix = str(prefix)

        self.prefix = str(prefix)

        kwargs['axes'] = x.axes
        super(PrintOp, self).__init__(args=(x,), **kwargs)

    def generate_adjoints(self, adjoints, delta, x):
        """
        adjoints pass through PrintOp unchanged.
        """
        x.generate_add_delta(adjoints, delta)
