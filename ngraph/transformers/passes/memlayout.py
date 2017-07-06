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

import copy
import six


from ngraph.transformers.passes.passes import GraphPass


class MemLayoutPass(GraphPass):
    def do_pass(self, computation_decl, **kwargs):
        self.exop_block = computation_decl.exop_block

        # this pass may be run multiple times
        # reset all of the allocated buffers to None before starting
        for exop in self.exop_block:
            for new in exop.liveness_new_list:
                new.buffer_pool_offset = None
            for free in exop.liveness_free_list:
                free.buffer_pool_offset = None

        # Layout temporary memory
        # self.layout_memory_middle_out()
        self.layout_memory_best_fit()

        # Layout persistent memory
        pmm = MemoryManager()
        for exop in self.exop_block:
            for input_decl in exop.input_decls:
                if input_decl.source_output_decl.tensor_decl.is_persistent and \
                        input_decl.source_output_decl.tensor_decl.buffer_pool_offset is None:
                    input_decl.source_output_decl.tensor.buffer_pool_offset = \
                        pmm.allocate(input_decl.source_output_decl.tensor.size)
            for output_decl in exop.output_decls:
                if output_decl.tensor_decl.is_persistent and \
                        output_decl.tensor_decl.buffer_pool_offset is None:
                    output_decl.tensor_decl.buffer_pool_offset = \
                        pmm.allocate(output_decl.tensor_decl.size)

        # self.test_memory_overlap()

    def layout_memory_best_fit(self):
        mm = MemoryManager()
        for i, node in enumerate(self.exop_block):
            for new in node.liveness_new_list:
                if new.buffer_pool_offset is not None:
                    raise RuntimeError('Error: {} - {} Already allocated'.format(i, new))
                else:
                    new.buffer_pool_offset = mm.allocate(new.size)

            for free in node.liveness_free_list:
                if free.buffer_pool_offset is None:
                    raise RuntimeError('Error: {} - {} Already free'.format(
                        i,
                        free.tensor_description_base.name))
                else:
                    mm.free(free.buffer_pool_offset)

    def layout_memory_first_fit(self):
        mm = MemoryManager()
        for i, node in enumerate(self.exop_block):
            for new in node.liveness_new_list:
                if new.buffer_pool_offset is not None:
                    raise RuntimeError('Error: {} - {} Already allocated'.format(i, new))
                else:
                    new.buffer_pool_offset = mm.allocate(new.size)

            for free in node.liveness_free_list:
                if free.buffer_pool_offset is None:
                    raise RuntimeError('Error: {} - {} Already free'.format(
                        i,
                        free.tensor_description_base.name))
                else:
                    mm.free(free.buffer_pool_offset)

    def layout_memory_middle_out(self):
        mm = MemoryManager()
        max_usage = 0
        max_op = None
        for op in self.exop_block:
            usage = op.memory_usage()
            if usage >= max_usage:
                max_usage = usage
                max_op = op
        # print('max op {}'.format(max_op))
        mm = MemoryManager()
        current_live = max_op.liveness_live_list
        for live in current_live:
            live.buffer_pool_offset = mm.allocate(live.size)

        # need to copy the state of the memory manager since the first half
        # of the algorithm allocates backwards in time. The second half needs
        # to pick up where the first half started, not where it ended
        mm2 = copy.deepcopy(mm)

        # first half
        # start at the op which uses the most memory and work
        # backwards to the beginning
        next_free_list = []
        node = max_op
        while not node.is_exop_end_of_list:
            live_list = node.liveness_live_list
            # free_list = [x for x in current_live if x not in live_list]
            free_list = next_free_list
            next_free_list = node.liveness_new_list
            current_live = live_list
            for free in free_list:
                mm.free(free.buffer_pool_offset)
            for live in current_live:
                if live.buffer_pool_offset is None:
                    live.buffer_pool_offset = mm.allocate(live.size)
            # node.validate()
            node = node.prev_exop

        # second half
        # now go forward from the next op after the max op
        mm = mm2
        node = max_op.next_exop
        current_live = node.liveness_live_list
        while not node.is_exop_end_of_list:
            for new in node.liveness_new_list:
                if new.buffer_pool_offset is None:
                    new.buffer_pool_offset = mm.allocate(new.size)
            for free in node.liveness_free_list:
                mm.free(free.buffer_pool_offset)
            # node.validate()
            node = node.next_exop

    def test_memory_overlap(self):
        for i, node in enumerate(self.exop_block):
            for tensor1 in node.liveness_live_list:
                for tensor2 in node.liveness_live_list:
                    if tensor1 != tensor2:
                        t1_start = tensor1.buffer_pool_offset
                        t2_start = tensor2.buffer_pool_offset
                        t1_end = t1_start + tensor1.size
                        t2_end = t2_start + tensor2.size
                        if t1_start < t2_start:
                            if t1_end > t2_start:
                                print('{} {}'.format(i, node.name))
                                print('   overlap {} at {} and {} at {}'.format(
                                    tensor1.size,
                                    tensor1.buffer_pool_offset,
                                    tensor2.size,
                                    tensor2.buffer_pool_offset))
                        else:
                            if t2_end > t1_start:
                                print('{} {}'.format(i, node.name))
                                print('   overlap {} at {} and {} at {}'.format(
                                    tensor1.size,
                                    tensor1.buffer_pool_offset,
                                    tensor2.size,
                                    tensor2.buffer_pool_offset))


class MemoryNode(object):
    def __init__(self, size, is_free=True):
        self.size = size
        self.is_free = is_free


class MemoryManager(object):
    '''
    All code here translated directly from NervanaSystems:memlayout c++ implementation by rhk
    '''

    def __init__(self, alignment=1):
        self.alignment = alignment
        self.node_list = [MemoryNode(six.MAXSIZE)]
        self.max_allocation = 0

    def __repr__(self):
        offset = 0
        res = []
        for i, node in enumerate(self.node_list):
            res.append('{}@{}{}'.format(node.size, offset, 'F' if node.is_free else 'A'))
            offset += node.size
        return " ".join(res)

    @staticmethod
    def align(size, alignment):
        return - (-size // alignment) * alignment

    def free(self, offset):
        found = False
        search_offset = 0
        found_index = 0
        for index, node in enumerate(self.node_list):
            if offset == search_offset:
                found_index = index
                found = True
                break
            else:
                search_offset += node.size

        if not found:
            raise RuntimeError("Offset {} not found".format(offset))

        if found_index > 0 and self.node_list[found_index - 1].is_free:
            self.node_list[found_index].size += self.node_list.pop(found_index - 1).size
            found_index -= 1

        if found_index < len(self.node_list) - 1 and self.node_list[found_index + 1].is_free:
            self.node_list[found_index].size += self.node_list.pop(found_index + 1).size

        self.node_list[found_index].is_free = True

    def allocate(self, size):
        return self.allocate_best_fit(size)
        # return self.allocate_first_fit(size)

    def allocate_first_fit(self, size):
        size = MemoryManager.align(size, self.alignment)
        offset = 0
        for i, node in enumerate(self.node_list):
            if node.is_free:
                if node.size == size:
                    node.is_free = False
                    break
                elif node.size > size:
                    self.node_list[i].size -= size
                    self.node_list.insert(i, MemoryNode(size, is_free=False))
                    break
            offset += node.size
        return offset

    def allocate_best_fit(self, size):
        size = MemoryManager.align(size, self.alignment)
        best_node, best_offset, best_delta = None, None, six.MAXSIZE
        offset = 0
        for i, node in enumerate(self.node_list):
            delta = node.size - size
            if node.is_free and delta >= 0 and delta < best_delta:
                best_i, best_node, best_offset, best_delta = i, node, offset, delta
            offset += node.size

        if not best_node:
            raise RuntimeError("Bad Allocation")
        else:
            if best_delta == 0:
                best_node.is_free = False
            else:
                self.node_list[best_i].size -= size
                self.node_list.insert(best_i, MemoryNode(size, is_free=False))

        self.max_allocation = max(self.max_allocation, best_offset + size)
        return best_offset

    def max_allocated(self):
        return self.max_allocation
