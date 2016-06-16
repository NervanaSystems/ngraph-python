from geon.backends.graph.names import NameableValue, NameScope

# TODO These are stubs for implementing Neon's layers

class Layer(object):
    def __init__(self, graph=None, parallelism="Unknown", **kargs):
        super(Layer, self).__init__(**kargs)

    def configure(self, graph, in_obj):
        """
        Add to computation graph for the layer.
        :param graph: The naming context.
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
        pass


class Convolution(ParameterLayer):
    def __init__(self, fshape, strides={}, padding={}, bsum=False, **kargs):
        super(Convolution, self).__init__(**kargs)


class Deconvolution(ParameterLayer):
    def __init__(self, fshape, strides={}, padding={}, bsum=False, **kargs):
        super(Deconvolution, self).__init__(**kargs)


class Linear(ParameterLayer):
    def __init__(self, nout, init, bsum=False, **kargs):
        super(Linear, self).__init__(**kargs)


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
        super(Bias, self).__init__(init, name)
        self.y = None
        self.owns_output = False
        self.owns_delta = False

    def __str__(self):
        if len(self.in_shape) == 3:
            layer_string = "Bias Layer '%s': size %d x (%dx%d)" % (
                self.name, self.in_shape[0], self.in_shape[1], self.in_shape[2])
        else:
            layer_string = "Bias Layer '%s': size %d" % (self.name, self.bias_size)
        return layer_string

    def configure(self, graph, in_obj):
        """
        Sets shape based parameters of this layer given an input tuple or int
        or input layer.

        Arguments:
            in_obj (int, tuple, Layer or Tensor): object that provides shape
                                                  information for layer

        Returns:
            (tuple): shape of output data
        """
        in_obj = super(Bias, self).configure(graph, in_obj)
        bias = ast.Parameter(axes=sample_axes(in_obj))
        axes(in_obj)
        self.out_shape = self.in_shape
        self.bias_size = self.in_shape[0]
        if self.weight_shape is None:
            self.weight_shape = (self.bias_size, 1)
        return self


class Activation(Layer):
    def __init__(self, transform, **kargs):
        super(Activation, self).__init__(**kargs)


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
    def __init__(self, bias=None, batch_norm=False, activation=None, name=None):
        if batch_norm and (bias is not None):
            raise AttributeError('Batchnorm and bias cannot be combined')
        self.activation = activation
        self.batch_norm = batch_norm
        self.bias = bias
        self.base_name = name

    def init_base_name(self):
        if self.base_name is None:
            self.base_name = self[-1].name

    def add_postfilter_layers(self):
        self.init_basename()
        if self.bias is not None:
            name = self.base_name + '_bias'
            self.append(Bias(init=self.bias, name=name))
        if self.batch_norm:
            name = self.base_name + '_bnorm'
            self.append(BatchNorm(name=name))
        if self.activation is not None:
            name = self.base_name + '_' + self.activation.classnm
            self.append(Activation(transform=self.activation, name=name))


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
                 batch_norm=False, activation=None, name=None):
        super(Affine, self).__init__(bias=bias, batch_norm=batch_norm,
                                     activation=activation, name=name)
        self.append(Linear(nout, init, bsum=batch_norm, name=name))
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


class BatchNorm(Layer):
    def __init__(self, rho=0.9, eps=1e-3, **kargs):
        super(BatchNorm, self).__init__(**kargs)


class BatchNormAutodiff(BatchNorm):
    def __init__(self, rho=0.99, eps=1e-6, **kargs):
        super(BatchNormAutodiff, self).__init__(**kargs)