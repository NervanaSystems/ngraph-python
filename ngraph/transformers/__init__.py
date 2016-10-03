from ngraph.transformers.base import set_transformer_factory, make_transformer
from ngraph.transformers.nptransform import NumPyTransformerFactory

__all__ = [set_transformer_factory, make_transformer]

set_transformer_factory(NumPyTransformerFactory())
