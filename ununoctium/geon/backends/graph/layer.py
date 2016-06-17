from geon.backends.graph.names import NameableValue, NameScope
import geon.backends.graph.funs as be

# TODO These are stubs for implementing Neon's layers

class Layer(object):
    def __init__(self, name=None, graph=None, axes=None, parallelism="Unknown", **kargs):
        super(Layer, self).__init__(**kargs)
        self.name = name
        self.axes = axes

    def configure(self, in_obj):
        """
        Add to computation graph for the layer.
        :param in_obj: The input for the layer
        :return: The output of the layer
        """
        return in_obj

class BranchNode(Layer):
    def __init__(self, **kargs):
        super(BranchNode, self).__init__(**kargs)


class SkipNode(Layer):
    def __init__(self, **kargs):
        super(SkipNode, self).__init__(**kargs)


class Pooling(Layer):
    def __init__(self, fshape, op="max", strides={}, padding={}, **kargs):
        super(Pooling, self).__init__(**kargs)


class ParameterLayer(Layer):
    def __init__(self, init=None, **kargs):
        super(ParameterLayer, self).__init__(**kargs)
        self.has_params = True
        self.init = init
        self.W = None
        self.dW = None
        self.batch_sum = None


class Convolution(ParameterLayer):
    def __init__(self, fshape, strides={}, padding={}, bsum=False, **kargs):
        super(Convolution, self).__init__(**kargs)


class Deconvolution(ParameterLayer):
    def __init__(self, fshape, strides={}, padding={}, bsum=False, **kargs):
        super(Deconvolution, self).__init__(**kargs)


class Linear(ParameterLayer):
    def __init__(self, nout, bsum=False, **kargs):
        super(Linear, self).__init__(**kargs)
        self.nout = nout
        self.inputs = None
        self.bsum = bsum

    def configure(self, in_obj):
        """
        Sets shape based parameters of this layer given an input tuple or int
        or input layer.

        Arguments:
            in_obj (int, tuple, Layer or Tensor): object that provides shape
                                                  information for layer

        Returns:
            (Tensor): output
        """
        in_obj = super(Linear, self).configure(in_obj)

        v = be.Variable(axes=be.linear_map_axes(be.sample_axes(in_obj),
                                                self.axes or [be.Axis(self.nout, name='Hidden')]),
                        init=self.init)
        return be.dot(v, in_obj)


class Bias(ParameterLayer):
    """
    A bias layer implemented that adds a learned bias to inputs and produces
    outputs of the same shape.

    Arguments:
        init (Initializer, optional): Initializer object to use for
            initializing layer bias
        name (str, optional): Layer name. Defaults to "BiasLayer"
    """

    def __init__(self, init, **kargs):
        super(Bias, self).__init__(**kargs)
        self.y = None
        self.owns_output = False
        self.owns_delta = False

    def configure(self, graph, in_obj):
        """
        Sets shape based parameters of this layer given an input tuple or int
        or input layer.

        Arguments:
            in_obj (int, tuple, Layer or Tensor): object that provides shape
                                                  information for layer

        Returns:
            (Tensor): output
        """
        in_obj = super(Bias, self).configure(graph, in_obj)
        return in_obj + be.Variable(axes=be.sample_axes(in_obj))


class Activation(Layer):
    """
    A layer that applies a specified transform to the inputs and
    produces outputs of the same shape.

    Generally used to implemenent nonlinearities for layer post activations.

    Arguments:
        transform (Transform): a transform object with fprop and bprop
            functions to apply
        name (str, optional): Layer name. Defaults to "ActivationLayer"
    """
    def __init__(self, transform, **kargs):
        super(Activation, self).__init__(**kargs)
        self.transform = transform

    def configure(self, in_obj):
        """
        Sets shape based parameters of this layer given an input tuple or int
        or input layer.

        Arguments:
            in_obj : input to the layer

        Returns:
            (Tensor): output
        """
        in_obj = super(Activation, self).configure(in_obj)
        return self.transform(in_obj)


class DataTransform(Layer):
    def __init__(self, transform, **kargs):
        super(DataTransform, self).__init__(**kargs)


class ColorNoise(Layer):
    def __init__(self, colorpca=None, colorstd=None, noise_coeff=0.1, name="ColorNoiseLayer", **kargs):
        super(ColorNoise, self).__init__(name=name, **kargs)


class CompoundLayer(list):
    """
    Base class for macro layers.
    """
    def __init__(self, bias=None, batch_norm=False, activation=None, name=None, axes=None):
        if batch_norm and (bias is not None):
            raise AttributeError('Batchnorm and bias cannot be combined')
        self.activation = activation
        self.batch_norm = batch_norm
        self.bias = bias
        self.axes = axes

    def add_postfilter_layers(self):
        if self.bias is not None:
            self.append(Bias(init=self.bias))
        if self.batch_norm:
            self.append(BatchNorm())
        if self.activation is not None:
            self.append(Activation(transform=self.activation))


class Affine(CompoundLayer):

    """
    A linear layer with a learned bias and activation, implemented as a list
    composing separate linear, bias/batchnorm and activation layers.

    Arguments:
        nout (int, tuple): Desired size or shape of layer output
        init (Initializer, optional): Initializer object to use for
            initializing layer weights and bias
        bias (Initializer): an initializer to use for bias parameters
        activation (Transform): a transform object with fprop and bprop
            functions to apply
        name (str): the root name for the layer, suffixes are automatically
            generated for the component layers

    """

    def __init__(self, nout, init, bias=None,
                 batch_norm=False, activation=None, name=None, **kargs):
        super(Affine, self).__init__(bias=bias, batch_norm=batch_norm,
                                     activation=activation, name=name, **kargs)
        self.append(Linear(nout, init=init, bsum=batch_norm, name=name, axes=self.axes))
        self.add_postfilter_layers()


class Conv(CompoundLayer):
    def __init__(self, fshape, init, strides={}, padding={}, **kargs):
        super(Conv, self).__init__(**kargs)


class Deconv(CompoundLayer):
    def __init__(self, fshape, init, strides={}, padding={}, **kargs):
        super(Deconv, self).__init__(**kargs)


class LRN(Layer):
    def __init__(self, depth, alpha=1., beta=0., ascale=1., bpower=1., **kargs):
        super(LRN, self).__init__(**kargs)


class Dropout(Layer):
    def __init__(self, keep=0.5, **kargs):
        super(Dropout, self).__init__(**kargs)


class LookupTable(ParameterLayer):
    def __init__(self, vocab_size, embedding_dim, init, update=True,
                 pad_idx=None, **kargs):
        super(LookupTable, self).__init__(**kargs)


class GeneralizedCost(object):

    """
    A cost layer that applies the provided cost function and computes errors
    with respect to inputs and targets.

    Arguments:
       costfunc (Cost): class with costfunc that computes errors
    """

    def __init__(self, costfunc, name=None, **kargs):
        super(GeneralizedCost, self).__init__(**kargs)
        self.costfunc = costfunc
        self.name = name

    def get_cost(self, inputs, targets):
        """
        Compute the cost function over the inputs and targets.

        Arguments:
            inputs (Tensor): Tensor containing input values to be compared to
                targets
            targets (Tensor): Tensor containing target values.

        Returns:
            Tensor containing cost

        """
        return self.costfunc(inputs, targets)


class BatchNorm(Layer):
    def __init__(self, rho=0.9, eps=1e-3, **kargs):
        super(BatchNorm, self).__init__(**kargs)


class BatchNormAutodiff(BatchNorm):
    def __init__(self, rho=0.99, eps=1e-6, **kargs):
        super(BatchNormAutodiff, self).__init__(**kargs)
