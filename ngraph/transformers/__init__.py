from ngraph.transformers.base import set_transformer_factory, make_transformer
from ngraph.transformers.nptransform import NumPyTransformerFactory
from ngraph.transformers.gputransform import GPUTransformerFactory

__all__ = [set_transformer_factory, make_transformer, NumPyTransformerFactory,
           GPUTransformerFactory]

set_transformer_factory(NumPyTransformerFactory())
