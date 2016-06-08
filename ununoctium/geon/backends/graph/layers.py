

# TODO These are stubs for implementing Neon's layers

class Layer(object):
    def __init__(self, name=None, parallelism="Unknown"):
        pass


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
    def __init__(self, init, **kargs):
        super(Bias, self).__init__(**kargs)


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
    def __init__(self, bias=None, batch_norm=False, activation=None, name=None):
        pass


class Affine(CompoundLayer):
    def __init__(self, nout, init, **kargs):
        super(Affine, self).__init__(**kargs)


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