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
from ngraph.op_graph.op_graph import OrderedSet


def test_empty_ordered_set():
    """
    Test if initialization fails if "values" is not a sequence
    """
    with pytest.raises(ValueError):
        OrderedSet(0)


def test_empty_update_set():
    """
    Test if update fails if "values" is not a sequence
    """
    testset = OrderedSet([1, 2, 3])
    with pytest.raises(ValueError):
        testset.update(values=0)


def test_reversed():
    ls = [1, 2, 3]
    testset = OrderedSet(ls)
    a = list(testset.__reversed__())
    b = list(reversed(ls))
    assert a == b


def test_remove():
    testset = OrderedSet([1, 2, 3])
    testset.remove(2)
    assert testset.elt_list == [1, 3]


def test_get_item():
    testset = OrderedSet(values=[1, 2, 3])
    assert testset[1] == 2


def test___add__():
    ls = [4, 5]
    testset1 = OrderedSet([1, 2, 3])
    testset2 = OrderedSet([1, 2, 3])
    testset1.update(ls)
    testset2 = testset2 + ls
    assert testset1 == testset2


def test_clear_set():
    testset = OrderedSet(values=[1, 2, 3])
    testset.clear()
    assert testset.__len__() == 0


def test_union():
    testset = OrderedSet([1, 2, 3])
    result = testset.union([-1, 1, 2])
    assert testset.elt_list == [1, 2, 3]
    assert result.elt_list == [1, 2, 3, -1]

    testset = OrderedSet([1, 2, 3])
    result = [-1, 1, 2] | testset
    assert testset.elt_list == [1, 2, 3]
    assert result.elt_list == [-1, 1, 2, 3]

    testset = OrderedSet([1, 2, 3])
    testset |= [-1, 1, 2]
    assert testset.elt_list == [1, 2, 3, -1]


def test_intersection():
    testset = OrderedSet([1, 2, 3])
    result = testset & [-1, 1, 2]
    assert testset.elt_list == [1, 2, 3]
    assert result.elt_list == [1, 2]

    testset = OrderedSet([1, 2, 3])
    result = [-1, 1, 2] & testset
    assert testset.elt_list == [1, 2, 3]
    assert result.elt_list == [1, 2]

    testset = OrderedSet([1, 2, 3])
    testset &= [-1, 1, 2]
    assert testset.elt_list == [1, 2]


def test_difference():
    testset = OrderedSet([1, 2, 3])
    result = testset - [-1, 1, 2]
    assert testset.elt_list == [1, 2, 3]
    assert result.elt_list == [3]

    testset = OrderedSet([1, 2, 3])
    result = [-1, 1, 2] - testset
    assert testset.elt_list == [1, 2, 3]
    assert result.elt_list == [-1]

    testset = OrderedSet([1, 2, 3])
    testset -= [-1, 1, 2]
    assert testset.elt_list == [3]
