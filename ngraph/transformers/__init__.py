from ngraph.transformers.base import Transformer
from ngraph.transformers.nptransform import NumPyTransformerFactory
from ngraph.transformers.gputransform import GPUTransformerFactory

__all__ = [Transformer, NumPyTransformerFactory, GPUTransformerFactory]

Transformer.set_transformer_factory(NumPyTransformerFactory())
