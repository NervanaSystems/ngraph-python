
from ngraph.frontends.neon import (Sequential, Deconvolution, Convolution, Rectlin, Affine,
                                   Tanh, GaussianInit)
import ngraph as ng

filter_init = GaussianInit(var=0.02)
relu = Rectlin(slope=0)
# leaky_relu
lrelu = Rectlin(slope=0.2)


# generator network
def make_generator_gp(bn=True, n_extra_layers=0, bias_init=None):
    deconv_layers = [Deconvolution((4, 4, 512), filter_init, strides=1, padding=0,
                                   activation=relu, batch_norm=bn, bias_init=bias_init),
                     Deconvolution((4, 4, 256), filter_init, strides=2, padding=1,
                                   activation=relu, batch_norm=bn, bias_init=bias_init),
                     Deconvolution((4, 4, 128), filter_init, strides=2, padding=1,
                                   activation=relu, batch_norm=bn, bias_init=bias_init),
                     Deconvolution((4, 4, 64), filter_init, strides=2, padding=1,
                                   activation=relu, batch_norm=bn, bias_init=bias_init)]

    for i in range(n_extra_layers):
        deconv_layers.append(Convolution((3, 3, 64), filter_init, strides=1, padding=1,
                                         activation=lrelu, batch_norm=bn, bias_init=bias_init))

    deconv_layers.append(Deconvolution((4, 4, 3), filter_init, strides=2, padding=1,
                                       activation=Tanh(), batch_norm=False, bias_init=bias_init))
    return Sequential(deconv_layers, name="Generator")


def make_generator(bn=True, bias_init=None):
    deconv_layers = [Affine(weight_init=filter_init, activation=None, batch_norm=False,
                            axes=ng.make_axes({"C": 1024, "H": 4, "W": 4})),
                     Deconvolution((4, 4, 512), filter_init, strides=2, padding=1,
                                   activation=relu, batch_norm=bn, bias_init=bias_init),
                     Deconvolution((4, 4, 256), filter_init, strides=2, padding=1,
                                   activation=relu, batch_norm=bn, bias_init=bias_init),
                     Deconvolution((4, 4, 128), filter_init, strides=2, padding=1,
                                   activation=relu, batch_norm=bn, bias_init=bias_init)
                     ]

    deconv_layers.append(Deconvolution((4, 4, 3), filter_init, strides=2, padding=1,
                                       activation=Tanh(), batch_norm=False, bias_init=bias_init))
    return Sequential(deconv_layers, name="Generator")


# discriminator network
def make_discriminator_gp(bn=True, n_extra_layers=0, disc_activation=None, bias_init=None):
    conv_layers = [Convolution((4, 4, 64), filter_init, strides=2, padding=1,
                               activation=lrelu, batch_norm=False, bias_init=bias_init)]

    for i in range(n_extra_layers):
        conv_layers.append(Convolution((3, 3, 64), filter_init, strides=1, padding=1,
                                       activation=lrelu, batch_norm=bn, bias_init=bias_init))

    conv_layers.append(Convolution((4, 4, 128), filter_init, strides=2, padding=1,
                                   activation=lrelu, batch_norm=bn, bias_init=bias_init))
    conv_layers.append(Convolution((4, 4, 256), filter_init, strides=2, padding=1,
                                   activation=lrelu, batch_norm=bn, bias_init=bias_init))
    conv_layers.append(Convolution((4, 4, 512), filter_init, strides=2, padding=1,
                                   activation=lrelu, batch_norm=bn, bias_init=bias_init))
    conv_layers.append(Convolution((4, 4, 1), filter_init, strides=1, padding=0,
                                   activation=disc_activation, batch_norm=False,
                                   bias_init=bias_init))
    return Sequential(conv_layers, name="Discriminator")


def make_discriminator(bn=True, disc_activation=None, bias_init=None):
    conv_layers = [Convolution((4, 4, 128), filter_init, strides=2, padding=1,
                               activation=lrelu, batch_norm=False, bias_init=bias_init)]

    conv_layers.append(Convolution((4, 4, 256), filter_init, strides=2, padding=1,
                                   activation=lrelu, batch_norm=bn, bias_init=bias_init))
    conv_layers.append(Convolution((4, 4, 512), filter_init, strides=2, padding=1,
                                   activation=lrelu, batch_norm=bn, bias_init=bias_init))
    conv_layers.append(Convolution((4, 4, 1024), filter_init, strides=2, padding=1,
                                   activation=lrelu, batch_norm=bn, bias_init=bias_init))
    conv_layers.append(Affine(weight_init=filter_init, activation=None, batch_norm=False,
                              axes=ng.make_axes({"C": 1, "H": 1, "W": 1})))
    return Sequential(conv_layers, name="Discriminator")
