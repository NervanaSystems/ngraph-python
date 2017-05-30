# ----------------------------------------------------------------------------
# Copyright 2015-2016 Nervana Systems Inc.
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
'''
Test generic Layer properties
'''
import ngraph as ng
from ngraph.frontends.neon import Layer, wrap_layer


class SimpleLayer(Layer):

    @wrap_layer(cache_key=Layer.inference_mode_key)
    def __call__(self, in_obj):
        if Layer.inference_mode:
            return in_obj
        else:
            return 2 * in_obj


def test_layer_caching():

    in_obj = ng.placeholder(())
    layer = SimpleLayer()
    out_train = layer(in_obj)
    out_train2 = layer(in_obj)
    with Layer.inference_mode_on():
        out_inference = layer(in_obj)
        out_inference2 = layer(in_obj)
    out_train3 = layer(in_obj)

    assert out_train is out_train2, "Training mode call not cached"
    assert out_inference is out_inference2, "Inference mode call not cached"
    assert out_train is not out_inference, "Training and inference mode calls are the same"
    assert out_train is out_train3, "Training mode not restored"
