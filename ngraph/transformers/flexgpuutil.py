from ngraph.transformers.gputransform import ElementWiseKernel
from ngraph.transformers.gpu.gemm import GEMMKernel
from ngraph.transformers.gpu.conv import ConvFpropKernel, ConvBpropKernel, ConvUpdateKernel
from ngraph.transformers.gputransform import FillKernel, DimShuffleKernel

def bind_flex_params(kernel):
    """
    bind flex scale kernel parameters
    ElementWiseKernel is defined in gputransform, this function avoids
    adding flex specific code there.
    """
    if isinstance(kernel, ElementWiseKernel):
        for index, flex_scale_desc in kernel.flex_scale_info:
            scale = flex_scale_desc.flex_entry.scale
            scale = 1.0/scale if flex_scale_desc.is_output else scale
            kernel.params[index] = scale
    # TODO: when all cases are covered:
    #elif hasattr(kernel, 'bind_flex_scales'): kernel.bind_flex_scales()
    elif isinstance(kernel, (FillKernel, DimShuffleKernel)):
        # TODO: check this is correct
        pass
    elif isinstance(kernel, GEMMKernel) or \
         isinstance(kernel, (ConvFpropKernel, ConvBpropKernel, ConvUpdateKernel)):
        kernel.bind_flex_scales()
    else:
        # TODO: handle other cases?
        print 'bind_flex_params unhandled kernel type {}'.format(kernel)
        raise NotImplementedError
