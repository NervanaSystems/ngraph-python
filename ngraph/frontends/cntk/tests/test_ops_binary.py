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


def test_plus_1():
    cntk_op = C.plus([1, 2, 3], [4, 5, 6])
    cntk_ret = cntk_op.eval()

    ng_op, _ = CNTKImporter().import_model(cntk_op)
    ng_ret = ng.transformers.make_transformer().computation(ng_op)()

    assert np.array_equal(cntk_ret, ng_ret)


def test_plus_2():
    cntk_op = C.plus([[1, 2, 3], [4, 5, 6]], [7, 8, 9])
    cntk_ret = cntk_op.eval()

    ng_op, _ = CNTKImporter().import_model(cntk_op)
    ng_ret = ng.transformers.make_transformer().computation(ng_op)()

    assert np.array_equal(cntk_ret, ng_ret)


def test_plus_3():
    cntk_op = C.plus([1, 2, 3], [[4, 5, 6], [7, 8, 9]])
    cntk_ret = cntk_op.eval()

    ng_op, _ = CNTKImporter().import_model(cntk_op)
    ng_ret = ng.transformers.make_transformer().computation(ng_op)()

    assert np.array_equal(cntk_ret, ng_ret)


def test_minus_1():
    cntk_op = C.minus([1, 2, 3], [4, 5, 6])
    cntk_ret = cntk_op.eval()

    ng_op, _ = CNTKImporter().import_model(cntk_op)
    ng_ret = ng.transformers.make_transformer().computation(ng_op)()

    assert np.array_equal(cntk_ret, ng_ret)


def test_minus_2():
    cntk_op = C.minus([[1, 2, 3], [4, 5, 6]], [7, 8, 9])
    cntk_ret = cntk_op.eval()

    ng_op, _ = CNTKImporter().import_model(cntk_op)
    ng_ret = ng.transformers.make_transformer().computation(ng_op)()

    assert np.array_equal(cntk_ret, ng_ret)


def test_minus_3():
    cntk_op = C.minus([1, 2, 3], [[4, 5, 6], [7, 8, 9]])
    cntk_ret = cntk_op.eval()

    ng_op, _ = CNTKImporter().import_model(cntk_op)
    ng_ret = ng.transformers.make_transformer().computation(ng_op)()

    assert np.array_equal(cntk_ret, ng_ret)


def test_element_times_1():
    cntk_op = C.element_times([1, 2, 3], [4, 5, 6])
    cntk_ret = cntk_op.eval()

    ng_op, _ = CNTKImporter().import_model(cntk_op)
    ng_ret = ng.transformers.make_transformer().computation(ng_op)()

    assert np.array_equal(cntk_ret, ng_ret)


def test_element_times_2():
    cntk_op = C.element_times([[1, 2, 3], [4, 5, 6]], [7, 8, 9])
    cntk_ret = cntk_op.eval()

    ng_op, _ = CNTKImporter().import_model(cntk_op)
    ng_ret = ng.transformers.make_transformer().computation(ng_op)()

    assert np.array_equal(cntk_ret, ng_ret)


def test_element_times_3():
    cntk_op = C.element_times([1, 2, 3], [[4, 5, 6], [7, 8, 9]])
    cntk_ret = cntk_op.eval()

    ng_op, _ = CNTKImporter().import_model(cntk_op)
    ng_ret = ng.transformers.make_transformer().computation(ng_op)()

    assert np.array_equal(cntk_ret, ng_ret)


def test_times_1():
    cntk_op = C.times([1, 2, 3], [[4], [5], [6]])
    cntk_ret = cntk_op.eval()

    ng_op, _ = CNTKImporter().import_model(cntk_op)
    ng_ret = ng.transformers.make_transformer().computation(ng_op)()

    assert np.array_equiv(cntk_ret, ng_ret)


def test_times_2():
    cntk_op = C.times([[1, 2], [3, 4]], [[5, 6], [7, 8]])
    cntk_ret = cntk_op.eval()

    ng_op, _ = CNTKImporter().import_model(cntk_op)
    ng_ret = ng.transformers.make_transformer().computation(ng_op)()

    assert np.array_equiv(cntk_ret, ng_ret)


def test_times_3():
    cntk_op = C.times([1, 2, 3], [[4, 5], [6, 7], [8, 9]])
    cntk_ret = cntk_op.eval()

    ng_op, _ = CNTKImporter().import_model(cntk_op)
    ng_ret = ng.transformers.make_transformer().computation(ng_op)()

    assert np.array_equiv(cntk_ret, ng_ret)


def test_times_4():
    cntk_op = C.times([[1, 2, 3], [4, 5, 6]], [[7], [8], [9]])
    cntk_ret = cntk_op.eval()

    ng_op, _ = CNTKImporter().import_model(cntk_op)
    ng_ret = ng.transformers.make_transformer().computation(ng_op)()

    assert np.array_equiv(cntk_ret, ng_ret)


def test_times_5():
    cntk_op = C.times([[1, 2, 3], [4, 5, 6]], [[7, 8], [9, 10], [11, 12]])
    cntk_ret = cntk_op.eval()

    ng_op, _ = CNTKImporter().import_model(cntk_op)
    ng_ret = ng.transformers.make_transformer().computation(ng_op)()

    assert np.array_equiv(cntk_ret, ng_ret)


def test_times_6():
    cntk_op = C.times([[1, 2], [3, 4], [5, 6]], [[7, 8, 9], [10, 11, 12]])
    cntk_ret = cntk_op.eval()

    ng_op, _ = CNTKImporter().import_model(cntk_op)
    ng_ret = ng.transformers.make_transformer().computation(ng_op)()

    assert np.array_equiv(cntk_ret, ng_ret)


if __name__ == "__main__":
    test_plus_1()
    test_plus_2()
    test_plus_3()
    test_minus_1()
    test_minus_2()
    test_minus_3()
    test_element_times_1()
    test_element_times_2()
    test_element_times_3()
    test_times_1()
    test_times_2()
    test_times_3()
    test_times_4()
    test_times_5()
    test_times_6()
