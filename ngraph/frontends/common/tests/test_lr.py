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

from __future__ import print_function, division
from ngraph.frontends.common.learning_rate_policies import lr_policies
import ngraph as ng
import numpy as np
from ngraph.testing import ExecutorFactory


def test_fixed_lr():
    # set up
    params = dict([('name', 'fixed'), ('max_iter', 20), ('base_lr', 0.9)])
    iter_buf = ng.placeholder(axes=(), dtype=np.dtype(np.uint64))

    # execute
    naive_lr = np.full(params['max_iter'], params['base_lr'])
    lr_op = lr_policies[params['name']]['obj'](params).compute_lr(iter_buf)
    with ExecutorFactory() as ex:
        compute_lr = ex.executor(lr_op, iter_buf)
        ng_lr = [compute_lr(i).item(0) for i in range(params['max_iter'])]

    # compare
        assert(np.allclose(ng_lr, naive_lr, atol=1e-4, rtol=1e-3))


def test_step_lr():
    # set up
    params = dict([('name', 'step'), ('max_iter', 20), ('base_lr', 0.9), ('gamma', 0.9),
                   ('step', 5)])
    iter_buf = ng.placeholder(axes=(), dtype=np.dtype(np.uint64))

    # execute
    naive_lr = params['base_lr'] * params['gamma'] ** \
               (np.array(range(0, params['max_iter'])) // params['step'])
    lr_op = lr_policies[params['name']]['obj'](params).compute_lr(iter_buf)
    with ExecutorFactory() as ex:
        compute_lr = ex.executor(lr_op, iter_buf)
        ng_lr = [compute_lr(i).item(0) for i in range(params['max_iter'])]

    # compare
        assert(np.allclose(ng_lr, naive_lr, atol=1e-4, rtol=1e-3))


def test_exp_lr():
    # set up
    params = dict([('name', 'exp'), ('max_iter', 20), ('base_lr', 0.9), ('gamma', 0.9)])
    iter_buf = ng.placeholder(axes=(), dtype=np.dtype(np.uint64))

    # execute
    naive_lr = params['base_lr'] * params['gamma'] ** np.array(range(0, params['max_iter']))
    lr_op = lr_policies[params['name']]['obj'](params).compute_lr(iter_buf)
    with ExecutorFactory() as ex:
        compute_lr = ex.executor(lr_op, iter_buf)
        ng_lr = [compute_lr(i).item(0) for i in range(params['max_iter'])]

    # compare
        assert(np.allclose(ng_lr, naive_lr, atol=1e-4, rtol=1e-3))


def test_inv_lr():
    # set up
    params = dict([('name', 'inv'), ('max_iter', 20), ('base_lr', 0.9), ('gamma', 0.9),
                  ('power', 0.75)])
    iter_buf = ng.placeholder(axes=(), dtype=np.dtype(np.uint64))

    # execute
    naive_lr = params['base_lr'] * (1 + params['gamma'] * np.array(range(0, params['max_iter']))) \
               ** (-params['power'])
    lr_op = lr_policies[params['name']]['obj'](params).compute_lr(iter_buf)
    with ExecutorFactory() as ex:
        compute_lr = ex.executor(lr_op, iter_buf)
        ng_lr = [compute_lr(i).item(0) for i in range(params['max_iter'])]

    # compare
        assert(np.allclose(ng_lr, naive_lr, atol=1e-4, rtol=1e-3))


def test_poly_lr():
    # set up
    params = dict([('name', 'poly'), ('max_iter', 20), ('base_lr', 0.9), ('power', 0.75)])
    iter_buf = ng.placeholder(axes=(), dtype=np.dtype(np.uint64))

    # execute
    naive_lr = params['base_lr'] * \
               (1 - np.array(range(0, params['max_iter'])) / params['max_iter']) ** params['power']
    lr_op = lr_policies[params['name']]['obj'](params).compute_lr(iter_buf)
    with ExecutorFactory() as ex:
        compute_lr = ex.executor(lr_op, iter_buf)
        ng_lr = [compute_lr(i).item(0) for i in range(params['max_iter'])]

    # compare
        assert(np.allclose(ng_lr, naive_lr, atol=1e-4, rtol=1e-3))


def test_sigmoid_lr():
    # set up
    params = dict([('name', 'sigmoid'), ('max_iter', 20), ('base_lr', 0.9), ('gamma', 0.75),
                   ('step_size', 5)])
    iter_buf = ng.placeholder(axes=(), dtype=np.dtype(np.uint64))

    # execute
    naive_lr = params['base_lr'] * (1 / (1 + np.exp(-params['gamma'] * (np.array(range(0, params['max_iter'])) - params['step_size']))))
    lr_op = lr_policies[params['name']]['obj'](params).compute_lr(iter_buf)
    with ExecutorFactory() as ex:
        compute_lr = ex.executor(lr_op, iter_buf)
        ng_lr = [compute_lr(i).item(0) for i in range(params['max_iter'])]

        # compare
        assert (np.allclose(ng_lr, naive_lr, atol=1e-4, rtol=1e-3))

