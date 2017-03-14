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
    parser.addoption("--batch_size", type=int, default=8,
                     help="Batch size for tests using input_tensor fixture.")
    parser.addoption("--transformer", default="numpy", choices=ngt.transformer_choices(),
                     help="Select from available transformers")

def pytest_xdist_node_collection_finished(node, ids):
    ids.sort()


@pytest.fixture(scope="module")
def transformer_factory(request):
    def set_and_get_factory(transformer_name):
        factory = ngt.make_transformer_factory(transformer_name)
        ngt.set_transformer_factory(factory)
        return factory

    name = request.config.getoption("--transformer")

    yield set_and_get_factory(name)

    # Reset transformer factory to default
    ngt.set_transformer_factory(ngt.make_transformer_factory("numpy"))
