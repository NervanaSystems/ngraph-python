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

from __future__ import division
from ngraph.transformers.passes.passes import GraphPass


class VisualizeMemPass(GraphPass):

    def __init__(self, filename='mem.html'):
        self.filename = filename

    def do_pass(self, computation_decl, **kwargs):
        self.computation_decl = computation_decl
        with open(self.filename, 'w') as file:
            file.truncate()
            file.write('<!DOCTYPE html>\n<html>\n')
            file.write('<head>\n')
            file.write('    <style>\n')
            file.write('        th, td {\n')
            file.write('            border-bottom: 1px solid #ddd;\n')
            file.write('            width: 200px;\n')
            file.write('        }\n')
            file.write('        table, td, th {\n')
            # file.write('            border: 1px solid #ddd;\n')
            # file.write('            text-align: left;\n')
            file.write('        }\n')
            file.write('        table {\n')
            file.write('            border-collapse: collapse;\n')
            # file.write('            width: 100%;\n')
            file.write('        }\n')
            # file.write('        tr:hover {background-color: #f5f5f5}\n')
            file.write('        tr:nth-child(even) {background-color: #f2f2f2}\n')
            file.write('    </style>\n')
            file.write('</head>\n')

            file.write('<body>\n')
            tensors = set()
            temp_max_size = 0
            for node in computation_decl.exop_block:
                tensors |= set(node.liveness_live_list)
            for tensor in tensors:
                if tensor.is_persistent is False:
                    temp_max_size += tensor.size

            file.write('<table>\n')
            file.write(
                '<tr><td>Persistent Memory Footprint</td><td align="right">{:,}</td></tr>\n'
                .format(computation_decl.exop_block.persistent_size()))
            file.write(
                '<tr><td>Temporary Memory Footprint</td><td align="right">{:,}</td></tr>\n'
                .format(computation_decl.exop_block.memory_footprint()))
            file.write(
                '<tr><td>Max temporary Memory Footprint</td><td align="right">{:,}</td></tr>\n'
                .format(temp_max_size))
            file.write('</table>\n')

            file.write('<hr>\n')
            self.draw_tensor_weight(file)
            # file.write('<hr>\n')
            # self.draw_op_influence(file)
            file.write('<hr>\n')
            self.draw_histogram(file)
            # file.write('<hr>\n')
            file.write('</body>\n</html>\n')

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

    def draw_tensor_weight(self, file):
        largest_op = self.find_largest_op()

        if largest_op is not None:
            largest_live = set()
            for tensor in largest_op.liveness_live_list:
                largest_live.add(tensor)

            age_list = dict()
            tensor_set = set()
            generator_op = dict()
            file.write('<table>\n')
            file.write('    <tr>')
            file.write('<th align="left">tensor</th>')
            file.write('<th align="right">size</th>')
            file.write('<th align="right">age</th>')
            file.write('<th align="right">generator weight</th>')
            file.write('</tr>\n')
            for i, exop in enumerate(self.computation_decl.exop_block):
                for tensor in exop.liveness_new_list:
                    age_list[tensor] = i
                    generator_op[tensor] = exop
                for tensor in exop.liveness_free_list:
                    start = age_list[tensor]
                    age_list[tensor] = (i - start)
                    tensor_set.add(tensor)
            for tensor in sorted(list(tensor_set), reverse=True, key=lambda tensor: tensor.size):
                generator_weight = self.compute_op_weight(generator_op[tensor])
                if tensor in largest_live:
                    file.write('    <tr style="background-color: #f0c0f0">')
                else:
                    file.write('    <tr>')
                file.write('<td>{}</td>'.format(tensor.tensor_description_base.name))
                file.write('<td align="right">{:,}</td>'.format(tensor.size))
                file.write('<td align="right">{}</td>'.format(age_list[tensor]))
                file.write('<td align="right">{}</td>'.format(generator_weight))
                file.write('</tr>\n')

            file.write('</table>\n')

    def draw_histogram(self, file):
        stroke_width = 14
        text_offset = 4
        offset = 200
        width = 1000
        scale = width - offset
        line_spacing = stroke_width * 1.5
        line_count = 0
        for _ in self.computation_decl.exop_block:
            line_count += 1
        height = line_count * line_spacing + stroke_width
        memory_footprint = max(1, float(self.computation_decl.exop_block.memory_footprint()))

        file.write('<svg viewBox="0 0 {} {}">\n'.format(width, height))
        y = 0
        for i, node in enumerate(self.computation_decl.exop_block):
            usage = float(node.memory_usage())
            footprint = float(node.memory_footprint())
            y += line_spacing
            x1 = offset
            x2 = ((usage / memory_footprint) * scale) + offset
            file.write('<text x="{}" y="{}" fill="{}">{}</text>\n'.format(
                0, y + text_offset, "black", node.name
            ))
            file.write('<line x1="{}" y1="{}" x2="{}" y2="{}"'
                       ' style="stroke:{};stroke-width:{}" />\n'
                       .format(x1, y, x2, y, "forestgreen", stroke_width))
            x1 = x2
            x2 = ((footprint / memory_footprint) * scale) + offset
            file.write('<line x1="{}" y1="{}" x2="{}" y2="{}"'
                       ' style="stroke:{};stroke-width:{}" />\n'
                       .format(x1, y, x2, y, "firebrick", stroke_width))
        file.write('</svg>\n')

    def draw_op_influence(self, file):
        file.write('<table>\n')
        file.write('    <tr>')
        file.write('<th align="left">op</th>')
        file.write('<th align="right">influence</th>')
        file.write('</tr>\n')
        for exop in self.computation_decl.exop:
            weight = self.compute_op_weight(exop)
            file.write('    <tr>')
            file.write('<td>{}</td>'.format(exop.name))
            file.write('<td align="right">{:,}</td>'.format(weight))
            file.write('</tr>\n')

    def compute_op_weight(self, exop):
        mass = 0
        # for input_decl in exop.input_decls:
        #     tensor = input_decl.source_output_decl.tensor
        #     if tensor.is_persistent is False:
        #         mass += tensor.size
        # for output_decl in exop.output_decls:
        #     tensor = output_decl.tensor
        #     if tensor.is_persistent is False:
        #         mass -= tensor.size
        for tensor in exop.liveness_new_list:
            if tensor.is_persistent is False:
                mass += tensor.size
        for tensor in exop.liveness_free_list:
            if tensor.is_persistent is False:
                mass -= tensor.size
        return mass
