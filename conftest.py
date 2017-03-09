# ----------------------------------------------------------------------------
# Copyright 2016 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
import pytest
import ngraph.transformers as ngt
from ngraph.flex.names import flex_gpu_transformer_name


def pytest_addoption(parser):
    parser.addoption("--enable_flex", action="store_true",
                     help="Enable and *only* enable {} transformer.".format(flex_gpu_transformer_name))
    parser.addoption("--enable_mkl", action="store_true",
                     help="Enable and *only* enable CPU MKL transformer.")
    parser.addoption("--batch_size", type=int, default=8,
                     help="Enable and *only* enable CPU MKL transformer.")


def pytest_xdist_node_collection_finished(node, ids):
    ids.sort()

@pytest.fixture(scope="module",
                params=ngt.transformer_choices())
def transformer_factory(request):
    def set_and_get_factory(transformer_name):
        factory = ngt.make_transformer_factory(transformer_name)
        ngt.set_transformer_factory(factory)
        return factory

    transformer_name = request.param

    if pytest.config.getoption("--enable_flex"):
        if transformer_name == flex_gpu_transformer_name:
            if flex_gpu_transformer_name in ngt.transformer_choices():
                yield set_and_get_factory(transformer_name)
            else:
                raise ValueError("GPU not found, should not set --enable_flex"
                                 "flag for py.test.")
        else:
            pytest.skip('Skip all other transformers since --enable_flex is set.')
    elif pytest.config.getoption("--enable_mkl"):
        if transformer_name == "cpu":
            yield set_and_get_factory(transformer_name)
        else:
            pytest.skip('Skip non cpu transformers since --enable_mkl is set.')
    else:
        if transformer_name == flex_gpu_transformer_name:
            pytest.skip('Skip flex test since --enable_flex is not set.')
        else:
            yield set_and_get_factory(transformer_name)

    # Reset transformer factory to default
    ngt.set_transformer_factory(ngt.make_transformer_factory("numpy"))
