import pytest
import ngraph.transformers as ngt


@pytest.fixture(scope="module",
                params=ngt.Transformer.transformer_choices())
def transformer_factory(request):
    factory = ngt.Transformer.make_transformer_factory(request.param)
    ngt.Transformer.set_transformer_factory(factory)
    yield factory

    # Reset transformer factory to default
    ngt.Transformer.set_transformer_factory(ngt.Transformer.make_transformer_factory("numpy"))
