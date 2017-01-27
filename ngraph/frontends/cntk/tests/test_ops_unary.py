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
import numpy as np

import ngraph as ng
from ngraph.frontends.cntk.cntk_importer.importer import CNTKImporter


def test_sigmoid_1():
    cntk_op = C.sigmoid([-2, -1., 0., 1., 2.])
    cntk_ret = cntk_op.eval()

    ng_op, _ = CNTKImporter().import_model(cntk_op)
    ng_ret = ng.transformers.make_transformer().computation(ng_op)()

    assert np.isclose(cntk_ret, ng_ret).all()


def test_sigmoid_2():
    cntk_op = C.sigmoid([0.])
    cntk_ret = cntk_op.eval()

    ng_op, _ = CNTKImporter().import_model(cntk_op)
    ng_ret = ng.transformers.make_transformer().computation(ng_op)()

    assert np.isclose(cntk_ret, ng_ret).all()


def test_sigmoid_3():
    cntk_op = C.exp([-0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.])
    cntk_ret = cntk_op.eval()

    ng_op, _ = CNTKImporter().import_model(cntk_op)
    ng_ret = ng.transformers.make_transformer().computation(ng_op)()

    assert np.isclose(cntk_ret, ng_ret).all()


def test_exp_1():
    cntk_op = C.exp([-2, -1., 0., 1., 2.])
    cntk_ret = cntk_op.eval()

    ng_op, _ = CNTKImporter().import_model(cntk_op)
    ng_ret = ng.transformers.make_transformer().computation(ng_op)()

    assert np.isclose(cntk_ret, ng_ret).all()


def test_exp_2():
    cntk_op = C.exp([0.])
    cntk_ret = cntk_op.eval()

    ng_op, _ = CNTKImporter().import_model(cntk_op)
    ng_ret = ng.transformers.make_transformer().computation(ng_op)()

    assert np.isclose(cntk_ret, ng_ret).all()


def test_exp_3():
    cntk_op = C.exp([-0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.])
    cntk_ret = cntk_op.eval()

    ng_op, _ = CNTKImporter().import_model(cntk_op)
    ng_ret = ng.transformers.make_transformer().computation(ng_op)()

    assert np.isclose(cntk_ret, ng_ret).all()


def test_tanh_1():
    cntk_op = C.tanh([-2, -1., 0., 1., 2.])
    cntk_ret = cntk_op.eval()

    ng_op, _ = CNTKImporter().import_model(cntk_op)
    ng_ret = ng.transformers.make_transformer().computation(ng_op)()

    assert np.isclose(cntk_ret, ng_ret).all()


def test_tanh_2():
    cntk_op = C.tanh([0.])
    cntk_ret = cntk_op.eval()

    ng_op, _ = CNTKImporter().import_model(cntk_op)
    ng_ret = ng.transformers.make_transformer().computation(ng_op)()

    assert np.isclose(cntk_ret, ng_ret).all()


def test_tanh_3():
    cntk_op = C.tanh([-0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.])
    cntk_ret = cntk_op.eval()

    ng_op, _ = CNTKImporter().import_model(cntk_op)
    ng_ret = ng.transformers.make_transformer().computation(ng_op)()

    assert np.isclose(cntk_ret, ng_ret).all()


if __name__ == "__main__":
    test_sigmoid_1()
    test_sigmoid_2()
    test_sigmoid_3()
    test_exp_1()
    test_exp_2()
    test_exp_3()
    test_tanh_1()
    test_tanh_2()
    test_tanh_3()
