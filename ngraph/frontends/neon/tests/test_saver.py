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
"""
This test trains weight for a simple model on cifar10 and saves the trained weights
to a file. And the the weights are loaded in to fresh inference problem
built from the same model. Classification error on validation set should match end of
training result.
"""
import os
from contextlib import closing
import numpy as np
import ngraph as ng
from ngraph.frontends.neon import ArrayIterator  # noqa
from ngraph.frontends.neon import CIFAR10  # noqa
from ngraph.frontends.neon import Affine, Convolution, Sequential
from ngraph.frontends.neon import KaimingInit, Rectlin, Softmax
from ngraph.frontends.neon import ax
from ngraph.frontends.neon import Saver
from ngraph.frontends.neon import GradientDescentMomentum
from ngraph.frontends.neon import Layer
import ngraph.transformers as ngt
from tqdm import tqdm


def test_persistent_tensor():
    input_axes = ng.make_axes([
        ng.make_axis(10),
        ng.make_axis(3)
    ])
    bgr = ng.persistent_tensor(
        axes=input_axes,
        initial_value=np.array([113.9, 123.0, 125.3]))
    bgr_comp = ng.computation(bgr, "all")

    results = dict()
    weight_saver = Saver()
    with closing(ngt.make_transformer()) as transformer:
        bgr_func = transformer.add_computation(bgr_comp)
        weight_saver.setup_save(transformer=transformer, computation=bgr_comp)
        results['saved'] = bgr_func()
        weight_saver.save(filename="test_persistent_tensor")
    with closing(ngt.make_transformer()) as restore_transformer:
        bgr_refunc = restore_transformer.add_computation(bgr_comp)
        weight_saver.setup_restore(transformer=restore_transformer, computation=bgr_comp,
                                   filename="test_persistent_tensor")
        weight_saver.restore()
        results['restored'] = bgr_refunc()
    assert np.allclose(results['saved'], results['restored'], atol=0)


def test_variable():
    input_axes = ng.make_axes([
        ng.make_axis(10),
        ng.make_axis(3)
    ])
    var = ng.variable(axes=input_axes)
    var_read = ng.computation(var, "all")
    var_comp = ng.computation(ng.AssignOp(tensor=var, val=np.array([113.9, 123.0, 125.3])), "all")
    results = dict()
    weight_saver = Saver()
    with closing(ngt.make_transformer()) as transformer:
        var_func = transformer.add_computation(var_comp)
        weight_saver.setup_save(transformer=transformer, computation=var_comp)
        results['saved'] = var_func()
        weight_saver.save(filename="test_variable")
    with closing(ngt.make_transformer()) as restore_transformer:
        var_readfunc = restore_transformer.add_computation(var_read)
        weight_saver.setup_restore(transformer=restore_transformer, computation=var_read,
                                   filename="test_variable")
        weight_saver.restore()
        results['restored'] = var_readfunc()
    assert np.allclose(results['saved'], results['restored'], atol=0)


def haha_test_affine_with_batch_norm():
    pass
