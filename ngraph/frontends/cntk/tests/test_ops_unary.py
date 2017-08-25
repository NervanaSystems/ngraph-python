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


def test_relu_1():
    cntk_op = C.relu([-2, -1., 0., 1., 2.])
    cntk_ret = cntk_op.eval()

    ng_op, _ = CNTKImporter().import_model(cntk_op)
    ng_ret = ng.transformers.make_transformer().computation(ng_op)()

    assert np.array_equal(cntk_ret, ng_ret)


def test_relu_2():
    cntk_op = C.relu([0.])
    cntk_ret = cntk_op.eval()

    ng_op, _ = CNTKImporter().import_model(cntk_op)
    ng_ret = ng.transformers.make_transformer().computation(ng_op)()

    assert np.array_equal(cntk_ret, ng_ret)


def test_relu_3():
    cntk_op = C.relu([-0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.])
    cntk_ret = cntk_op.eval()

    ng_op, _ = CNTKImporter().import_model(cntk_op)
    ng_ret = ng.transformers.make_transformer().computation(ng_op)()

    assert np.array_equal(cntk_ret, ng_ret)


def test_relu_4():
    cntk_op = C.relu([[1, 2, 3], [4, 5, 6]])
    cntk_ret = cntk_op.eval()

    ng_op, _ = CNTKImporter().import_model(cntk_op)
    ng_ret = ng.transformers.make_transformer().computation(ng_op)()

    assert np.array_equal(cntk_ret, ng_ret)


def test_relu_5():
    cntk_op = C.relu([[-3, -2, -1], [1, 2, 3]])
    cntk_ret = cntk_op.eval()

    ng_op, _ = CNTKImporter().import_model(cntk_op)
    ng_ret = ng.transformers.make_transformer().computation(ng_op)()

    assert np.array_equal(cntk_ret, ng_ret)


def test_reciprocal_1():
    cntk_op = C.reciprocal([-1 / 3, 1 / 5, -2, 3])
    cntk_ret = cntk_op.eval()

    ng_op, _ = CNTKImporter().import_model(cntk_op)
    ng_ret = ng.transformers.make_transformer().computation(ng_op)()

    assert np.array_equal(cntk_ret, ng_ret)


def test_negate_1():
    cntk_op = C.negate([-1, 1, -2, 3])
    cntk_ret = cntk_op.eval()

    ng_op, _ = CNTKImporter().import_model(cntk_op)
    ng_ret = ng.transformers.make_transformer().computation(ng_op)()

    assert np.array_equal(cntk_ret, ng_ret)


def test_log_1():
    cntk_op = C.log([1., 2.])
    cntk_ret = cntk_op.eval()

    ng_op, _ = CNTKImporter().import_model(cntk_op)
    ng_ret = ng.transformers.make_transformer().computation(ng_op)()

    assert np.array_equal(cntk_ret, ng_ret)


def test_sqrt_1():
    cntk_op = C.sqrt([0., 4.])
    cntk_ret = cntk_op.eval()

    ng_op, _ = CNTKImporter().import_model(cntk_op)
    ng_ret = ng.transformers.make_transformer().computation(ng_op)()

    assert np.array_equal(cntk_ret, ng_ret)


def test_floor_1():
    cntk_op = C.floor([0.2, 1.3, 4., 5.5, 0.0])
    cntk_ret = cntk_op.eval()

    ng_op, _ = CNTKImporter().import_model(cntk_op)
    ng_ret = ng.transformers.make_transformer().computation(ng_op)()

    assert np.array_equal(cntk_ret, ng_ret)


def test_floor_2():
    cntk_op = C.floor([[0.6, 3.3], [1.9, 5.6]])
    cntk_ret = cntk_op.eval()

    ng_op, _ = CNTKImporter().import_model(cntk_op)
    ng_ret = ng.transformers.make_transformer().computation(ng_op)()

    assert np.array_equal(cntk_ret, ng_ret)


def test_floor_3():
    cntk_op = C.floor([-5.5, -4.2, -3., -0.7, 0])
    cntk_ret = cntk_op.eval()

    ng_op, _ = CNTKImporter().import_model(cntk_op)
    ng_ret = ng.transformers.make_transformer().computation(ng_op)()

    assert np.array_equal(cntk_ret, ng_ret)


def test_floor_4():
    cntk_op = C.floor([[-0.6, -4.3], [1.9, -3.2]])
    cntk_ret = cntk_op.eval()

    ng_op, _ = CNTKImporter().import_model(cntk_op)
    ng_ret = ng.transformers.make_transformer().computation(ng_op)()

    assert np.array_equal(cntk_ret, ng_ret)


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
    test_relu_1()
    test_relu_2()
    test_relu_3()
    test_relu_4()
    test_relu_5()
    test_reciprocal_1()
    test_negate_1()
    test_log_1()
    test_sqrt_1()
    test_floor_1()
    test_floor_2()
    test_floor_3()
    test_floor_4()
