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


#@pytest.fixture(scope="class", params=ngt.transformer_choices())
@pytest.fixture(scope="class", params=['numpy', 'gpu'])
def transformer_factory(request):
    factory = ngt.make_transformer_factory(request.param)
    ngt.set_transformer_factory(factory)
    yield factory

    # Reset transformer factory to default
    ngt.set_transformer_factory(
        ngt.make_transformer_factory("numpy"))
