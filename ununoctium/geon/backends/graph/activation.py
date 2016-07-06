import geon.backends.graph.funs as be


class Transform(object):
    def __init__(self, name=None):
        self.name = name


class Rectlin(Transform):
    """
    Rectified Linear Unit (ReLu) activation function, :math:`f(x) = \max(x, 0)`.
    Can optionally set a slope which will make this a Leaky ReLu.
    """

    def __init__(self, slope=0, **kargs):
        """
        Class constructor.

        Args:
            slope (float, optional): Slope for negative domain. Defaults to 0.
            name (string, optional): Name to assign this class instance.
        """
        super(Rectlin, self).__init__(**kargs)
        self.slope = slope

    def __call__(self, x):
        """
        Returns the Exponential Linear activation

        Arguments:
            x (Tensor or optree): Input value

        Returns:
            Tensor or optree: output activation
        """
        return be.maximum(x, 0) + self.slope * be.minimum(0, x)


class Identity(Transform):
    """
    Identity activation function, :math:`f(x) = x`
    """

    def __init__(self, **kargs):
        """
        Class constructor.
        """
        super(Identity, self).__init__(**kargs)

    def __call__(self, x):
        """
        Returns the input as output.

        Arguments:
            x (Tensor or optree): input value

        Returns:
            Tensor or optree: identical to input
        """
        return x


class Explin(Transform):
    """
    Exponential Linear activation function, :math:`f(x) = \max(x, 0) + \\alpha (e^{\min(x, 0)}-1)`

    From: Clevert, Unterthiner and Hochreiter, ICLR 2016.
    """

    def __init__(self, alpha=1.0, **kargs):
        """
        Class constructor.

        Arguments:
            alpha (float): weight of exponential factor for negative values (default: 1.0).
            name (string, optional): Name (default: None)
        """
        super(Explin, self).__init__(**kargs)
        self.alpha = alpha

    def __call__(self, x):
        """
        Returns the Exponential Linear activation

        Arguments:
            x (Tensor or optree): input value

        Returns:
            Tensor or optree: output activation
        """
        return be.maximum(x, 0) + self.alpha * (be.exp(be.minimum(x, 0)) - 1)


class Normalizer(Transform):
    """
    Normalize inputs by a fixed divisor.
    """

    def __init__(self, divisor=128., **kargs):
        """
        Class constructor.

        Arguments:
            divisor (float, optional): Normalization factor (default: 128)
            name (string, optional): Name (default: None)
        """
        super(Normalizer, self).__init__(**kargs)
        self.divisor = divisor

    def __call__(self, x):
        """
        Returns the normalized value.

        Arguments:
            x (Tensor or optree): Input value

        Returns:
            Tensor or optree: Output :math:`x / N`
        """
        return x / self.divisor


class Softmax(Transform):
    """
    SoftMax activation function. Ensures that the activation output sums to 1.
    """

    def __init__(self, epsilon=2 ** -23, **kargs):
        """
        Class constructor.

        Arguments:
            name (string, optional): Name (default: none)
            epsilon (float, optional): Not used.
        """
        super(Softmax, self).__init__(**kargs)

    def __call__(self, x):
        """
        Returns the Softmax value.

        Arguments:
            x (Tensor or optree): Input value

        Returns:
            Tensor or optree: Output activation
        """
        return be.softmax(x)


class Tanh(Transform):
    """
    Hyperbolic tangent activation function, :math:`f(x) = \\tanh(x)`.
    """

    def __init__(self, **kargs):
        """
        Class constructor.
        """
        super(Tanh, self).__init__(**kargs)

    def __call__(self, x):
        """
        Returns the hyperbolic tangent.

        Arguments:
            x (Tensor or optree): Input value

        Returns:
            Tensor or optree: Output activation
        """
        return be.tanh(x)


class Logistic(Transform):
    """
    Logistic sigmoid activation function, :math:`f(x) = 1 / (1 + \exp(-x))`

    Squashes the input from range :math:`[-\infty,+\infty]` to :math:`[0, 1]`
    """

    def __init__(self, shortcut=False, **kargs):
        """
        Initialize Logistic based on whether shortcut is True or False. Shortcut
        should be set to true when Logistic is used in conjunction with a CrossEntropy cost.
        Doing so allows a shortcut calculation to be used during backpropagation.

        Args:
            shortcut (bool): If True, shortcut calculation will be used during backpropagation.

        """
        super(Logistic, self).__init__(**kargs)

    def __call__(self, x):
        """
        Returns the sigmoidal activation.

        Arguments:
            x (Tensor or optree): Input value

        Returns:
            Tensor or optree: Output activation
        """
        return be.sig(x)
