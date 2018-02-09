# ******************************************************************************
# Copyright 2017-2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************
"""
This test tests weight saving and restoring for persistent_tensor
and variable. All saved and restored values should match.
"""
import os
from contextlib import closing
import numpy as np
import ngraph as ng
from ngraph.frontends.neon import Saver
import ngraph.transformers as ngt


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
        results['saved'] = bgr_func().copy()
        weight_saver.save(filename="test_persistent_tensor")
    with closing(ngt.make_transformer()) as restore_transformer:
        bgr_refunc = restore_transformer.add_computation(bgr_comp)
        weight_saver.setup_restore(transformer=restore_transformer, computation=bgr_comp,
                                   filename="test_persistent_tensor")
        weight_saver.restore()
        results['restored'] = bgr_refunc().copy()
    os.remove("test_persistent_tensor.npz")
    assert np.allclose(results['saved'], results['restored'], atol=0)


def test_variable():
    input_axes = ng.make_axes([
        ng.make_axis(10),
        ng.make_axis(3)
    ])
    var = ng.variable(axes=input_axes)
    assign_val = np.random.rand(10, 3)
    var_assign = ng.AssignOp(tensor=var, val=assign_val)
    var_seq = ng.sequential([var_assign, var])
    var_comp = ng.computation(var_seq, "all")
    results = dict()
    weight_saver = Saver()
    with closing(ngt.make_transformer()) as transformer:
        var_func = transformer.add_computation(var_comp)
        weight_saver.setup_save(transformer=transformer, computation=var_comp)
        results['saved'] = var_func().copy()
        weight_saver.save(filename="test_variable")

    reassign_val = np.random.rand(10, 3)
    var_reassign = ng.AssignOp(tensor=var, val=reassign_val)

    var_recomp = ng.computation(var_reassign, "all")
    var_read = ng.computation(var, "all")
    with closing(ngt.make_transformer()) as restore_transformer:
        var_recompfunc = restore_transformer.add_computation(var_recomp)
        weight_saver.setup_restore(transformer=restore_transformer, computation=var_recomp,
                                   filename="test_variable")
        var_readfunc = restore_transformer.add_computation(var_read)
        var_recompfunc()
        results['reassigned'] = var_readfunc().copy()
        weight_saver.restore()
        results['restored'] = var_readfunc().copy()
    os.remove("test_variable.npz")
    assert np.allclose(results['saved'], assign_val, atol=0)
    assert np.allclose(results['reassigned'], reassign_val, atol=0)
    assert np.allclose(results['saved'], results['restored'], atol=0)
