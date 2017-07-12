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


from ngraph.transformers.passes.passes import GraphPass


class VerifyPass(GraphPass):
    def do_pass(self, computation_decl, **kwargs):
        self.computation_decl = computation_decl

        # Make sure they can print. Since this is python there are no compile time checks.
        for exop in computation_decl.exop_block:
            str(exop)

        self.test_read_before_write()

    def test_read_before_write(self):
        written_tensors = set()
        for exop in self.computation_decl.exop_block:
            for arg in exop.args:
                tensor = arg.read_view.tensor
                if tensor.is_persistent is False:
                    if tensor not in written_tensors:
                        raise RuntimeError(
                            'tensor read before written: {} - {}'.format(exop.name, tensor))

            for output_decl in exop.output_decls:
                if output_decl.tensor_view_decl.tensor.is_persistent is False:
                    written_tensors.add(output_decl.tensor_view_decl.tensor)
