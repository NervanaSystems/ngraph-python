from geon.backends.graph.names import NameableValue, NameScope
import geon.backends.graph.funs as be
import geon.backends.graph.arrayaxes as arrayaxes

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

    """
    Convolutional layer implementation.

    Arguments:
        fshape (tuple(int)): three dimensional shape of convolution window
        strides (int, dict, optional): strides to apply convolution
            window over. An int applies to both dimensions, or a dict with
            str_h and str_w applies to h and w dimensions distinctly.  Defaults
            to str_w = str_h = None
        padding (int, dict, optional): padding to apply to edges of
            input. An int applies to both dimensions, or a dict with pad_h
            and pad_w applies to h and w dimensions distinctly.  Defaults
            to pad_w = pad_h = None
        init (Initializer, optional): Initializer object to use for
            initializing layer weights
        name (str, optional): layer name. Defaults to "ConvolutionLayer"
    """

    def __init__(self, fshape, strides={}, padding={}, init=None, bsum=False,
                 name=None, parallelism="Data", cafe_compat=False, dtype=None):
        super(Convolution, self).__init__(init, name, parallelism)
        self.nglayer = None
        self.bsum = bsum
        self.convparams = {'str_h': 1, 'str_w': 1, 'str_d': 1,
                           'pad_h': 0, 'pad_w': 0, 'pad_d': 0,
                           'T': 1, 'D': 1}  # 3D paramaters

        # keep around args in __dict__ for get_description.
        self.fshape = fshape
        self.strides = strides
        self.padding = padding
        self.cafe_compat = cafe_compat
        self.dtype = dtype

        if isinstance(fshape, tuple) or isinstance(fshape, list):
            fkeys = ('R', 'S', 'K') if len(fshape) == 3 else ('T', 'R', 'S', 'K')
            fshape = {k: x for k, x in zip(fkeys, fshape)}
        if isinstance(strides, int):
            strides = {'str_h': strides, 'str_w': strides}
        if isinstance(padding, int):
            padding = {'pad_h': padding, 'pad_w': padding}
        for d in [fshape, strides, padding]:
            self.convparams.update(d)

    def __str__(self):
        spatial_dim = len(self.in_shape[1:])
        spatial_str = "%d x (" + "x".join(("%d",) * spatial_dim) + ")"
        padstr_str = ",".join(("%d",) * spatial_dim)
        padstr_dim = ([] if spatial_dim == 2 else ['d']) + ['h', 'w']

        pad_tuple = tuple(self.convparams[k] for k in ['pad_' + d for d in padstr_dim])
        str_tuple = tuple(self.convparams[k] for k in ['str_' + d for d in padstr_dim])

        fmt_tuple = (self.name,) + self.in_shape + self.out_shape + pad_tuple + str_tuple
        fmt_string = "Convolution Layer '%s': " + \
                     spatial_str + " inputs, " + spatial_str + " outputs, " + \
                     padstr_str + " padding, " + padstr_str + " stride"

        return ((fmt_string % fmt_tuple))

    def configure(self, in_obj):
        """
        Sets shape based parameters of this layer given an input tuple or int
        or input layer.

        Arguments:
            in_obj (int, tuple, Layer or Tensor): object that provides shape
                                                  information for layer

        Returns:
            (tuple): shape of output data
        """
        super(Convolution, self).configure(in_obj)
        if self.nglayer is None:
            assert isinstance(self.in_shape, tuple)
            ikeys = ('C', 'H', 'W') if len(self.in_shape) == 3 else ('C', 'D', 'H', 'W')
            shapedict = {k: x for k, x in zip(ikeys, self.in_shape)}
            shapedict['N'] = self.be.bsz
            self.convparams.update(shapedict)
            self.nglayer = ConvLayer(self.dtype, **self.convparams)
            (K, M, P, Q, N) = self.nglayer.dimO
            self.out_shape = (K, P, Q) if M == 1 else (K, M, P, Q)
        if self.weight_shape is None:
            self.weight_shape = self.nglayer.dimF2  # (C * R * S, K)
        if self.bsum:
            self.batch_sum_shape = (self.nglayer.K, 1)
        return self

    def fprop(self, inputs, inference=False, beta=0.0):
        """
        Apply the forward pass transformation to the input data.

        Arguments:
            inputs (Tensor): input data
            inference (bool): is inference only
            beta (float, optional): scale to apply to the outputs

        Returns:
            Tensor: output data
        """
        self.inputs = inputs
        self.be.fprop_conv(self.nglayer, inputs, self.W, self.outputs, beta=beta,
                           bsum=self.batch_sum)
        return self.outputs

    def bprop(self, error, alpha=1.0, beta=0.0):
        """
        Apply the backward pass transformation to the input data.

        Arguments:
            error (Tensor): deltas back propagated from the adjacent higher layer
            alpha (float, optional): scale to apply to input for activation
                                     gradient bprop.  Defaults to 1.0
            beta (float, optional): scale to apply to output activation
                                    gradient bprop.  Defaults to 0.0

        Returns:
            Tensor: deltas to propagate to the adjacent lower layer
        """
        if self.deltas:
            self.be.bprop_conv(self.nglayer, self.W, error, self.deltas,
                               alpha=alpha, beta=beta)
        self.be.update_conv(self.nglayer, self.inputs, error, self.dW)
        return self.deltas


class ConvLayer(object):

    """
    ConvLayer parameter object.
    This then is passed as an argument to all the convolution operations.

    N: Number of images in mini-batch
    C: Number of input feature maps
    K: Number of output feature maps

    D: Depth  of input image
    H: Height of input image
    W: Width  of input image

    T: Depth  of filter kernel
    R: Height of filter kernel
    S: Width  of filter kernel

    padding: amount of zero-padding around the given edge
    strides: factor to step the filters by in a given direction
    """

    def __init__(self, dtype,
                 N, C, K,
                 D=1, H=1, W=1,
                 T=1, R=1, S=1,
                 pad_d=0, pad_h=0, pad_w=0,
                 str_d=1, str_h=1, str_w=1,
                 cafe_compatibility=False):
        self.cafe_compatibility = cafe_compatibility

        # Compute the output spatial dimensions
        M = arrayaxes.output_dim(D, T, pad_d, str_d, cafe_compatibility=cafe_compatibility)
        P = arrayaxes.output_dim(H, R, pad_h, str_h, cafe_compatibility=cafe_compatibility)
        Q = arrayaxes.output_dim(W, S, pad_w, str_w, cafe_compatibility=cafe_compatibility)

        self.C = C
        self.K = K
        self.M = M
        self.P = P
        self.Q = Q
        self.NCK = (N, C, K)
        self.TRS = (T, R, S)
        self.DHW = (D, H, W)
        self.MPQ = (M, P, Q)
        self.padding = (pad_d, pad_h, pad_w)
        self.strides = (str_d, str_h, str_w)

        self.dimI = (C, D, H, W, N)
        self.dimF = (C, T, R, S, K)
        self.dimO = (K, M, P, Q, N)
        self.dimS = (K, 1)
        self.dimI2 = (C * D * H * W, N)
        self.dimF2 = (C * T * R * S, K)
        self.dimO2 = (K * M * P * Q, N)
        self.sizeI = reduce(mul, self.dimI, 1)
        self.sizeF = reduce(mul, self.dimF, 1)
        self.sizeO = reduce(mul, self.dimO, 1)
        self.nOut = reduce(mul, self.MPQ, 1) * K

        if all(x == 1 for x in self.TRS) and \
           all(p == 0 for p in self.padding) and \
           all(s == 1 for s in self.strides):

            self.dot = True
        else:
            self.dot = False

            self.mSlice = [self.fprop_slice(m, T, D, pad_d, str_d) for m in range(M)]
            self.pSlice = [self.fprop_slice(p, R, H, pad_h, str_h) for p in range(P)]
            self.qSlice = [self.fprop_slice(q, S, W, pad_w, str_w) for q in range(Q)]

    def fprop_slice(self, q, S, X, padding, strides):
        firstF = 0
        lastF = S - 1
        qs = q * strides - padding
        x2 = qs + lastF
        if qs < 0:
            firstF = -qs
            qs = 0
        if x2 >= X:
            dif = x2 - X + 1
            lastF -= dif
            x2 -= dif
        return (slice(firstF, lastF + 1), slice(qs, x2 + 1), lastF - firstF + 1)

    #####

    def fprop_conv(self, I, F, O):

        if X is None:
            X = O

        I = I._tensor.reshape(self.dimI)
        O = O._tensor.reshape(self.dimO)
        F = F._tensor.reshape(self.dimF)
        if bsum is not None:
            bsum = bsum._tensor.reshape((O.shape[0], 1))

        # 1x1 conv can be cast as a simple dot operation
        if self.dot:
            C = F.shape[0]
            K = F.shape[-1]
            # KxHWN = CxK.T . CxHWN
            F = F.reshape((C, K)).T
            I = I.reshape((C, -1))

            O[:] = np.dot(F, I).reshape(O.shape)
            return

        mSlice, pSlice, qSlice = self.mSlice, self.pSlice, self.qSlice

        K, M, P, Q, N = O.shape

        for m in range(M):
            sliceT, sliceD, _ = mSlice[m]
            for p in range(P):
                sliceR, sliceH, _ = pSlice[p]
                for q in range(Q):
                    sliceS, sliceW, _ = qSlice[q]

                    slicedF = F[:, sliceT, sliceR, sliceS, :].reshape((-1, K))
                    slicedI = I[:, sliceD, sliceH, sliceW, :].reshape((-1, N))

                    O[:, m, p, q, :] = np.dot(slicedF.T, slicedI)

    #####

    def xprop_conv(self, I, F, O, X=None, bias=None, bsum=None, alpha=1.0, beta=0.0,
                   relu=False, brelu=False, slope=0.0, backward=False):

        if X is None:
            X = O

        if backward:
            I = I._tensor.reshape(self.dimO)
            O = O._tensor.reshape(self.dimI)
            X = X._tensor.reshape(self.dimI)
        else:
            I = I._tensor.reshape(self.dimI)
            O = O._tensor.reshape(self.dimO)
            X = X._tensor.reshape(self.dimO)
        F = F._tensor.reshape(self.dimF)
        if bias is not None:
            bias = bias._tensor.reshape((O.shape[0], 1))
        if bsum is not None:
            bsum = bsum._tensor.reshape((O.shape[0], 1))

        # 1x1 conv can be cast as a simple dot operation
        if self.dot:
            C = F.shape[0]
            K = F.shape[-1]
            if backward:
                # CxHWN = CxK . KxHWN
                F = F.reshape((C, K))
                I = I.reshape((K, -1))
            else:
                # KxHWN = CxK.T . CxHWN
                F = F.reshape((C, K)).T
                I = I.reshape((C, -1))

            if beta:
                O[:] = alpha * np.dot(F, I).reshape(O.shape) + beta * X
            else:
                O[:] = np.dot(F, I).reshape(O.shape)
                self.compound_ops(O, X, bias, bsum, relu, brelu, slope)
            return

        if backward:
            # C <=> K and mirror T, R, S  (0, 1, 2, 3, 4) => (4, 1, 2, 3, 0)
            F = np.transpose(F[:, ::-1, ::-1, ::-1, :], (4, 1, 2, 3, 0)).copy()
            mSlice, pSlice, qSlice = self.dSlice, self.hSlice, self.wSlice
        else:
            mSlice, pSlice, qSlice = self.mSlice, self.pSlice, self.qSlice

        K, M, P, Q, N = O.shape

        for m in range(M):
            sliceT, sliceD, _ = mSlice[m]
            for p in range(P):
                sliceR, sliceH, _ = pSlice[p]
                for q in range(Q):
                    sliceS, sliceW, _ = qSlice[q]

                    slicedF = F[:, sliceT, sliceR, sliceS, :].reshape((-1, K))
                    slicedI = I[:, sliceD, sliceH, sliceW, :].reshape((-1, N))

                    if beta:
                        O[:, m, p, q, :] = alpha * np.dot(slicedF.T, slicedI) + \
                            beta * X[:, m, p, q, :]
                    else:
                        O[:, m, p, q, :] = np.dot(slicedF.T, slicedI)

        if not beta:
            self.compound_ops(O, X, bias, bsum, relu, brelu, slope)

    def update_conv(self, I, E, U, alpha=1.0, beta=0.0):

        C = self.C
        K, M, P, Q, N = self.dimO

        I = I._tensor.reshape(self.dimI)
        E = E._tensor.reshape(self.dimO)
        U = U._tensor.reshape(self.dimF)

        # 1x1 conv can be cast as a simple dot operation
        if self.dot:
            # CxK = CxHWN . KxHWN.T
            I = I.reshape((C, -1))
            E = E.reshape((K, -1)).T
            if beta:
                U[:] = alpha * np.dot(I, E).reshape(U.shape) + beta * U
            else:
                U[:] = alpha * np.dot(I, E).reshape(U.shape)
            return

        if beta:
            U *= beta
        else:
            U.fill(0.0)

        for m in range(M):
            sliceT, sliceD, tlen = self.mSlice[m]
            for p in range(P):
                sliceR, sliceH, rlen = self.pSlice[p]
                for q in range(Q):
                    sliceS, sliceW, slen = self.qSlice[q]

                    slicedI = I[:, sliceD, sliceH, sliceW, :].reshape((-1, N))
                    slicedE = E[:, m, p, q, :]
                    update = np.dot(slicedI, slicedE.T).reshape((C, tlen, rlen, slen, K))
                    if alpha == 1.0:
                        U[:, sliceT, sliceR, sliceS, :] += update
                    else:
                        U[:, sliceT, sliceR, sliceS, :] += alpha * update


class DeconvLayer(ConvLayer):

    """
    DeconvLayer parameter object.
    This then is passed as an argument to all the convolution operations.

    N: Number of images in mini-batch
    C: Number of output feature maps
    K: Number of input feature maps

    P: Height of input
    Q: Width of input

    D: Depth  of output image
    H: Height of output image
    W: Width  of output image

    T: Depth  of filter kernel
    R: Height of filter kernel
    S: Width  of filter kernel

    padding: amount of zero-padding around the given edge
    strides: factor to step the filters by in a given direction
    """

    def __init__(self, lib, dtype,
                 N, C, K,
                 P, Q,
                 R=1, S=1,
                 pad_d=0, pad_h=0, pad_w=0,
                 str_d=1, str_h=1, str_w=1):

        # Set T, M and D to be consts.
        T = 1
        M = 1
        D = 1

        # Cannot get exact, e.g. because not unique
        H = (P - 1) * str_h - 2 * pad_h + R
        W = (Q - 1) * str_w - 2 * pad_w + S

        # Add below to get H and W tracked
        self.H = H
        self.W = W

        self.C = C
        self.K = K
        self.M = M
        self.P = P
        self.Q = Q
        self.NCK = (N, C, K)
        self.TRS = (T, R, S)
        self.DHW = (D, H, W)
        self.MPQ = (M, P, Q)
        self.padding = (pad_d, pad_h, pad_w)
        self.strides = (str_d, str_h, str_w)

        # Did not change the names of dimI, dimO, etc. even though dimI is now technically the
        # dimension of the output
        self.dimI = (C, D, H, W, N)
        self.dimF = (C, T, R, S, K)
        self.dimO = (K, M, P, Q, N)
        self.dimI2 = (C * D * H * W, N)
        self.dimF2 = (C * T * R * S, K)
        self.dimO2 = (K * M * P * Q, N)
        self.sizeI = reduce(mul, self.dimI, 1)
        self.sizeF = reduce(mul, self.dimF, 1)
        self.sizeO = reduce(mul, self.dimO, 1)
        # nOut has to change because P and Q are now the inputs
        self.nOut = reduce(mul, self.DHW, 1) * C

        if all(x == 1 for x in self.TRS) and \
           all(p == 0 for p in self.padding) and \
           all(s == 1 for s in self.strides):

            self.dot = True
        else:
            self.dot = False

            self.dSlice = [self.bprop_slice(d, T, M, pad_d, str_d) for d in range(D)]
            self.hSlice = [self.bprop_slice(h, R, P, pad_h, str_h) for h in range(H)]
            self.wSlice = [self.bprop_slice(w, S, Q, pad_w, str_w) for w in range(W)]
            self.mSlice = [self.fprop_slice(m, T, D, pad_d, str_d) for m in range(M)]
            self.pSlice = [self.fprop_slice(p, R, H, pad_h, str_h) for p in range(P)]
            self.qSlice = [self.fprop_slice(q, S, W, pad_w, str_w) for q in range(Q)]


class PoolLayer(object):

    """
    PoolLayer parameter object.
    This then is passed as an argument to all pooling kernels.

    op: max, avg, l2 pooling
    N: Number of images in mini-batch

    C: Number of input feature maps
    D: Depth  of input image
    H: Height of input image
    W: Width  of input image

    J: Size of feature map pooling window (maxout n_pieces)
    T: Depth  of pooling window
    R: Height of pooling window
    S: Width  of pooling window

    padding: amount of zero-padding around the given image or feature map edge
    strides: factor to step the window by in a given direction (overlap allowed)

    Leave spatial dimensions at 1 to allow feature map pooling in the fc layers.

    """

    def __init__(self, lib, dtype,
                 op, N, C,
                 D=1, H=1, W=1,
                 J=1, T=1, R=1, S=1,
                 pad_c=0, pad_d=0, pad_h=0, pad_w=0,
                 str_c=None, str_d=None, str_h=None, str_w=None):

        # default to non-overlapping
        if str_c is None:
            str_c = J
        if str_d is None:
            str_d = T
        if str_h is None:
            str_h = R
        if str_w is None:
            str_w = S

        if str_c < J or str_d < T or str_h < R or str_w < S:
            self.overlap = (math.ceil(float(J) / str_c) *
                            math.ceil(float(T) / str_d) *
                            math.ceil(float(R) / str_h) *
                            math.ceil(float(S) / str_w))
        else:
            self.overlap = 0.0

        # Compute the output dimensions
        K = lib.output_dim(C, J, pad_c, str_c, pooling=True)
        M = lib.output_dim(D, T, pad_d, str_d, pooling=True)
        P = lib.output_dim(H, R, pad_h, str_h, pooling=True)
        Q = lib.output_dim(W, S, pad_w, str_w, pooling=True)

        self.op = op
        self.C = C
        self.K = K
        self.M = M
        self.P = P
        self.Q = Q
        self.N = N
        self.JTRS = (J, T, R, S)
        self.DHW = (D, H, W)
        self.MPQ = (M, P, Q)
        self.padding = (pad_c, pad_d, pad_h, pad_w)
        self.strides = (str_c, str_d, str_h, str_w)

        self.dimI = (C, D, H, W, N)
        self.dimO = (K, M, P, Q, N)
        self.dimF2 = None
        self.dimI2 = (C * D * H * W, N)
        self.dimO2 = (K * M * P * Q, N)
        self.sizeI = reduce(mul, self.dimI, 1)
        self.sizeO = reduce(mul, self.dimO, 1)
        self.nOut = reduce(mul, self.MPQ, 1) * K

        self.kSlice = [self.pool_slice(k, J, C, pad_c, str_c) for k in range(K)]
        self.mSlice = [self.pool_slice(m, T, D, pad_d, str_d) for m in range(M)]
        self.pSlice = [self.pool_slice(p, R, H, pad_h, str_h) for p in range(P)]
        self.qSlice = [self.pool_slice(q, S, W, pad_w, str_w) for q in range(Q)]

    def pool_slice(self, q, S, X, padding, strides):
        qs = q * strides - padding
        firstI = None
        for s in range(S):
            x = qs + s
            if x >= 0 and x < X:
                if firstI is None:
                    firstI = x
                lastI = x
        return (slice(firstI, lastI + 1), lastI - firstI + 1)


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

        v = be.Variable(axes=be.linear_map_axes(be.sample_axes(in_obj.axes),
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

    """
    A convolutional layer with a learned bias and activation, implemented as a
    list composing separate Convolution, Bias and Activation layers.

    Arguments:
        fshape (tuple(int)): three dimensional shape of convolution window
        init (Initializer, optional): Initializer object to use for
            initializing layer weights and bias
        strides (int, dict, optional): strides to apply convolution
            window over. An int applies to both dimensions, or a dict with
            str_h and str_w applies to h and w dimensions distinctly.  Defaults
            to str_w = str_h = None
        pad (int, dict, optional): padding to apply to edges of
            input. An int applies to both dimensions, or a dict with pad_h
            and pad_w applies to h and w dimensions distinctly.  Defaults
            to pad_w = pad_h = None
        bias (Initializer): an initializer to use for bias parameters
        activation (Transform): a transform object with fprop and bprop
            functions to apply
        name (str): the root name for the layer, suffixes are automatically
            generated for the component layers

    """

    def __init__(self, fshape, init, strides={}, padding={},
                 bias=None,
                 batch_norm=False,
                 activation=None,
                 name=None):
        super(Conv, self).__init__(bias=bias, batch_norm=batch_norm,
                                   activation=activation, name=name)
        self.append(Convolution(fshape=fshape, strides=strides, padding=padding,
                                init=init, bsum=batch_norm,
                                name=name))
        self.add_postfilter_layers()


class Deconv(CompoundLayer):

    """
    Same as Conv layer, but implements a composite deconvolution layer.
    """

    def __init__(self, fshape, init, strides={}, padding={}, bias=None, batch_norm=False,
                 activation=None, name=None):
        super(Deconv, self).__init__(bias=bias, batch_norm=batch_norm,
                                     activation=activation, name=name)
        self.append(Deconvolution(fshape=fshape, strides=strides, padding=padding,
                                  init=init, bsum=batch_norm))
        self.add_postfilter_layers()


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
