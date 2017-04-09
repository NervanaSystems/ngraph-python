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


class OpsBridge():
    """
    Bridging op between Caffe2 / ngraph.
    """

    def __init__(self):
        self._funcs_class_map = {}
        self.register_funcs_with_op_class()

    def map_funcs_with_op_class(self,func,className):
        """
        This function builds dictionary for op functions and its class
        """
        if not self._funcs_class_map.has_key(func):
            self._funcs_class_map[func] = className
        else:
            raise  ValueError (func," already exists in the class:",className)

    def get_op_class_by_func(self,func):
        """
        returns the op class for a given op function
        """
        if self._funcs_class_map.has_key(func):
            return self._funcs_class_map[func]
        else:
            raise NotImplementedError (func," is not defined in any class")

    def register_funcs_with_op_class(self):
        """
        This function registers all the op functions of caffe to op class
        """
        from ngraph.frontends.caffe.cf_importer.ops_constant import OpsConstant
        from ngraph.frontends.caffe.cf_importer.ops_binary import OpsBinary

        self.map_funcs_with_op_class("Eltwise",OpsBinary)
        self.map_funcs_with_op_class("DummyData",OpsConstant)

    def __call__(self, layer, input_ops):
        """
        This function returns the ngraph op corresponding to caffe layer type

        Arguments:
            layer : a Caffe layer
            input_ops (List): list of ngraph ops

        Returns:
            The resulting ngraph op
        """
        func = layer.type
        op_class = self.get_op_class_by_func(func)
        return op_class()(func,layer,input_ops)
