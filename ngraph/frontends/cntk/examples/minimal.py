# ----------------------------------------------------------------------------
# Copyright 2017 Nervana Systems Inc.
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
from __future__ import print_function

import cntk as C

import ngraph as ng
from ngraph.frontends.cntk.cntk_importer.importer import CNTKImporter

cntk_op = C.minus([1, 2, 3], [4, 5, 6])

ng_op, ng_placeholders = CNTKImporter().import_model(cntk_op)
results = ng.transformers.make_transformer().computation(ng_op)
print(results())
