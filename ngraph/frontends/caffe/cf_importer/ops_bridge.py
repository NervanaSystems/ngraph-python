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

funcs_class_bridge = {}

def register_func_with_ops_bridge(func,class_obj):
    """
    This function registers the op function against its class to ops bridge
    """
    if not funcs_class_bridge.has_key(func):
        funcs_class_bridge[func] = class_obj
    else:
        raise  ValueError (func," already exists in the class:",class_obj.__class__.__name__)

class OpsBridge():
    """
    Bridging ops between Caffe2 and ngraph.
    """

    def __init__(self):
        pass

    def get_op_class_by_func(self,func):
        """
        returns the op class for a given op function
        """
        if funcs_class_bridge.has_key(func):
            return funcs_class_bridge[func]
        else:
            raise NotImplementedError (func," is not defined in any class")


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
        return op_class(func,layer,input_ops)
