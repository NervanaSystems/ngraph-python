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

import sys


class DumpGraphPass(object):
    def __init__(self, **kwargs):
        self.filename = kwargs.pop('filename', 'graph.txt')
        self.show_live = kwargs.pop('show_live', False)

    def wrapped_do_pass(self, computation_decl, **kwargs):
        self.computation_decl = computation_decl
        if self.filename is None:
            f = sys.stdout
        else:
            f = open(self.filename, 'w')

        largest_op = self.find_largest_op()

        tensors = list()
        for i, exop in enumerate(computation_decl.exop_block):
            for input_decl in exop.input_decls:
                if input_decl.tensor_decl not in tensors:
                    tensors.append(input_decl.tensor_decl)
            for output_decl in exop.output_decls:
                if output_decl.tensor_decl not in tensors:
                    tensors.append(output_decl.tensor_decl)
            f.write('{} {} {}\n'.format(
                i, exop.name, '************' if exop is largest_op else ''))
            f.write('\tinputs: {}\n'.format(
                ", ".join([self.tensor_name(x.tensor_decl) for x in exop.input_decls])))
            f.write('\toutputs: {}\n'.format(
                ", ".join([self.tensor_name(x.tensor_decl) for x in exop.output_decls])))
            if self.show_live:
                f.write('\tlive: {}\n'.format(
                    ", ".join([self.tensor_name(x) for x in exop.liveness_live_list])))
                f.write('\tnew: {}\n'.format(
                    ", ".join([self.tensor_name(x) for x in exop.liveness_new_list])))
                f.write('\tfree: {}\n'.format(
                    ", ".join([self.tensor_name(x) for x in exop.liveness_free_list])))

        # # compute tensor weight in byte*ops
        # age_list = dict()
        # weight_list = dict()
        # for i, exop in enumerate(computation.exop):
        #     for tensor in exop.liveness_new_list:
        #         age_list[tensor] = i
        #     for tensor in exop.liveness_free_list:
        #         start = age_list[tensor]
        #         weight_list[tensor] = (i-start)*tensor.size
        #         # print('tensor {} age {}, weight {:,}'.format(
        # tensor.tensor_description_base.name, i-start, weight_list[tensor]))

        # make a list of tensors
        for exop in computation_decl.exop_block:
            for obj in exop.input_decls + exop.output_decls:
                if obj.tensor_decl not in tensors:
                    tensors.append(obj.tensor_decl)

        f.write('\n')
        for tensor in sorted(tensors, key=lambda tensor: tensor.tensor_description_base.name):
            f.write('tensor {}{}{}{}{} {}\n'.format(
                'P' if tensor.is_persistent else ' ',
                'I' if tensor.is_input else ' ',
                'O' if tensor.is_output else ' ',
                'C' if tensor.is_constant else ' ',
                'c' if tensor.is_compile_only else ' ',
                tensor))

        if f is not sys.stdout:
            f.close()

    def tensor_name(self, tensor):
        return str(tensor)

    def find_largest_op(self):
        largest_op = None
        largest_size = 0
        for i, exop in enumerate(self.computation_decl.exop_block):
            size = 0
            for tensor in exop.liveness_live_list:
                size += tensor.size
            if size > largest_size:
                largest_size = size
                largest_op = exop
        return largest_op
