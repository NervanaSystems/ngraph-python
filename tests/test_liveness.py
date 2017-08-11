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
import pytest

import ngraph as ng
from ngraph.transformers.passes.memlayout import MemoryManager
from ngraph.testing import ExecutorFactory


def test_liveness():
    with ExecutorFactory() as ex:

        x = ng.variable(axes=[]).named('x')
        y = ng.variable(axes=[]).named('y')
        w1 = ng.variable(axes=[]).named('w1')
        w2 = ng.variable(axes=[]).named('w2')

        x2 = x * w1
        x3 = (x2 * w2).named('result')
        cost = x3 - y

        dw1 = ng.deriv(cost, w1)
        dw2 = ng.deriv(cost, w2)

        upd1 = ng.assign(w1, w1 + dw1)
        upd2 = ng.assign(w2, w2 + dw2)
        seq_stuff = ng.sequential([upd1, upd2, x3])

        exc = ex.executor(seq_stuff)
        return exc

        # lg = LivenessGraph(exc.exop.ops)
        # lg.layout_memory()

        # for i, node in enumerate(lg.liveness_nodes):
        #     print i, node

        # for node in lg.liveness_nodes:
        #     for var1 in node.live_list:
        #         assert var1.buffer_pool_offset is not None
        #         for var2 in node.live_list:
        #             if var1 != var2:
        #                 if var1.buffer_pool_offset < var2.buffer_pool_offset:
        #                     assert var1.buffer_pool_offset + var1.size <= var2.buffer_pool_offset
        #                 else:
        #                     assert var2.buffer_pool_offset + var2.size <= var1.buffer_pool_offset

        # # for o in egraph.computations:
        # #     print o.values

        # print("max memory {}".format(lg.memory_footprint()))
        # print("worst case memory {}".format(lg.worst_case_memory_usage()))
        # print("memory efficiency {}".format(lg.memory_efficiency()))
        # # # print lg.liveness_json()


def test_memory_manager_allocate():
    mm = MemoryManager()

    assert 0 == mm.allocate(10)
    assert 10 == mm.allocate(10)
    assert 20 == mm.allocate(10)


def test_memory_manager_free_first_allocated():
    mm = MemoryManager()

    assert 0 == mm.allocate(10)
    assert 10 == mm.allocate(10)
    assert 3 == len(mm.node_list)

    mm.free(0)

    assert 3 == len(mm.node_list)
    assert mm.node_list[0].is_free is True
    assert mm.node_list[1].is_free is False
    assert mm.node_list[2].is_free is True


def test_memory_manager_free_middle_allocated():
    mm = MemoryManager()

    assert 0 == mm.allocate(10)
    assert 10 == mm.allocate(10)
    assert 20 == mm.allocate(10)
    assert 30 == mm.allocate(10)
    assert 40 == mm.allocate(10)
    assert 6 == len(mm.node_list)

    mm.free(10)

    assert 6 == len(mm.node_list)
    assert mm.node_list[0].is_free is False
    assert mm.node_list[1].is_free is True
    assert mm.node_list[2].is_free is False
    assert mm.node_list[3].is_free is False
    assert mm.node_list[4].is_free is False


def test_memory_manager_free_last_allocated():
    mm = MemoryManager()

    assert 0 == mm.allocate(10)
    assert 10 == mm.allocate(10)
    assert 20 == mm.allocate(10)
    assert 30 == mm.allocate(10)
    assert 40 == mm.allocate(10)
    assert 6 == len(mm.node_list)

    mm.free(40)

    assert 5 == len(mm.node_list)
    assert mm.node_list[0].is_free is False
    assert mm.node_list[1].is_free is False
    assert mm.node_list[2].is_free is False
    assert mm.node_list[3].is_free is False
    assert mm.node_list[4].is_free is True


def test_memory_manager_free_first_free():
    mm = MemoryManager()

    assert 0 == mm.allocate(10)
    assert 10 == mm.allocate(10)
    assert 20 == mm.allocate(10)
    assert 30 == mm.allocate(10)
    assert 40 == mm.allocate(10)
    assert 6 == len(mm.node_list)

    mm.free(10)
    mm.free(0)

    assert 5 == len(mm.node_list)
    assert mm.node_list[0].is_free is True
    assert mm.node_list[1].is_free is False
    assert mm.node_list[2].is_free is False
    assert mm.node_list[3].is_free is False


def test_memory_manager_free_middle_free():
    mm = MemoryManager()

    assert 0 == mm.allocate(10)
    assert 10 == mm.allocate(10)
    assert 20 == mm.allocate(10)
    assert 30 == mm.allocate(10)
    assert 40 == mm.allocate(10)
    assert 6 == len(mm.node_list)

    mm.free(0)
    mm.free(20)
    mm.free(10)

    assert 4 == len(mm.node_list)
    assert mm.node_list[0].is_free is True
    assert mm.node_list[1].is_free is False
    assert mm.node_list[2].is_free is False


def test_memory_manager_max_allocated():
    mm = MemoryManager()

    assert 0 == mm.allocate(10)
    assert 10 == mm.allocate(10)
    assert 20 == mm.allocate(10)
    assert 30 == mm.allocate(10)
    assert 40 == mm.allocate(10)
    assert 6 == len(mm.node_list)

    mm.free(0)
    mm.free(20)
    mm.free(10)

    assert mm.max_allocated() == 50


def test_memory_manager_bad_free():
    mm = MemoryManager()

    with pytest.raises(RuntimeError):
        mm.free(10)


def test_memory_manager_align():
    assert 8 == MemoryManager.align(1, 8)
    assert 8 == MemoryManager.align(2, 8)
    assert 8 == MemoryManager.align(3, 8)
    assert 8 == MemoryManager.align(4, 8)
    assert 8 == MemoryManager.align(5, 8)
    assert 8 == MemoryManager.align(6, 8)
    assert 8 == MemoryManager.align(7, 8)
    assert 8 == MemoryManager.align(8, 8)
    assert 16 == MemoryManager.align(9, 8)


def test_memory_manager_memory_align():
    mm = MemoryManager(8)

    assert 0 == mm.allocate(4)
    assert 8 == mm.allocate(4)
    assert 16 == mm.allocate(4)

# import ptvsd
# ptvsd.enable_attach(secret='nervana', address = ('0.0.0.0', 8080))
# print('Waiting for debugger to attach...')
# ptvsd.wait_for_attach()
# print('attached')

# test_liveness()
# test_memory_manager_allocate()
# test_memory_manager_free_first_allocated()
# test_memory_manager_free_middle_allocated()
# test_memory_manager_free_last_allocated()
# test_memory_manager_free_first_free()
# test_memory_manager_free_middle_free()
# test_memory_manager_max_allocated()
# test_memory_manager_bad_free()
# test_memory_manager_align()
# test_memory_manager_memory_align()
