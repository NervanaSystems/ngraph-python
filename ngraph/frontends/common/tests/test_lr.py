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
import pytest

pytestmark = pytest.mark.transformer_dependent("module")


@pytest.fixture
def iter_buf():
    return ng.placeholder(axes=(), dtype=np.dtype(np.uint32))


@pytest.fixture
def base_lr():
    return 0.9


@pytest.fixture
def max_iter():
    return 20


def test_fixed_lr(iter_buf, max_iter, base_lr):
    # set up
    name = 'fixed'
    params = {'name': name,
              'max_iter': max_iter,
              'base_lr': base_lr}

    # execute
    naive_lr = np.full(max_iter, base_lr)
    lr_op = lr_policies[name]['obj'](params)(iter_buf)
    with ExecutorFactory() as ex:
        compute_lr = ex.executor(lr_op, iter_buf)
        ng_lr = [compute_lr(i).item(0) for i in range(max_iter)]

        # compare
        ng.testing.assert_allclose(ng_lr, naive_lr, atol=1e-4, rtol=1e-3)


def test_step_lr(iter_buf, max_iter, base_lr):
    # set up
    name = 'step'
    gamma = 0.9
    step = 5
    params = {'name': name,
              'max_iter': max_iter,
              'base_lr': base_lr,
              'gamma': gamma,
              'step': step}

    # execute
    naive_lr = base_lr * gamma ** (np.arange(max_iter) // step)
    lr_op = lr_policies[name]['obj'](params)(iter_buf)
    with ExecutorFactory() as ex:
        compute_lr = ex.executor(lr_op, iter_buf)
        ng_lr = [compute_lr(i).item(0) for i in range(max_iter)]

        # compare
        ng.testing.assert_allclose(ng_lr, naive_lr, atol=1e-4, rtol=1e-3)


def test_exp_lr(iter_buf, max_iter, base_lr):
    # set up
    name = 'exp'
    gamma = 0.9
    params = {'name': name,
              'max_iter': max_iter,
              'base_lr': base_lr,
              'gamma': gamma}

    # execute
    naive_lr = base_lr * gamma ** np.arange(max_iter)
    lr_op = lr_policies[name]['obj'](params)(iter_buf)
    with ExecutorFactory() as ex:
        compute_lr = ex.executor(lr_op, iter_buf)
        ng_lr = [compute_lr(i).item(0) for i in range(max_iter)]

        # compare
        ng.testing.assert_allclose(ng_lr, naive_lr, atol=1e-4, rtol=1e-3)


def test_inv_lr(iter_buf, max_iter, base_lr):
    # set up
    name = 'inv'
    gamma = 0.9
    power = 0.75
    params = {'name': name,
              'max_iter': max_iter,
              'base_lr': base_lr,
              'gamma': gamma,
              'power': power}

    # execute
    naive_lr = base_lr * (1 + gamma * np.arange(max_iter)) ** (-power)
    lr_op = lr_policies[name]['obj'](params)(iter_buf)
    with ExecutorFactory() as ex:
        compute_lr = ex.executor(lr_op, iter_buf)
        ng_lr = [compute_lr(i).item(0) for i in range(max_iter)]

        # compare
        ng.testing.assert_allclose(ng_lr, naive_lr, atol=1e-4, rtol=1e-3)


@pytest.mark.skip(reason="Results mismatch on almost all transformers")
def test_poly_lr(iter_buf, max_iter, base_lr):
    # set up
    name = 'poly'
    power = 0.75
    params = {'name': name,
              'max_iter': max_iter,
              'base_lr': base_lr,
              'power': power}

    # execute
    naive_lr = base_lr * (1 - np.arange(max_iter) / max_iter) ** power
    lr_op = lr_policies[name]['obj'](params)(iter_buf)
    with ExecutorFactory() as ex:
        compute_lr = ex.executor(lr_op, iter_buf)
        ng_lr = [compute_lr(i).item(0) for i in range(max_iter)]

        # compare
        ng.testing.assert_allclose(ng_lr, naive_lr, atol=1e-4, rtol=1e-3)


@pytest.mark.skip(reason="Results mismatch on almost all transformers")
def test_sigmoid_lr(iter_buf, max_iter, base_lr):
    # set up
    name = 'sigmoid'
    gamma = 0.75
    step_size = 5
    params = {'name': name,
              'max_iter': max_iter,
              'base_lr': base_lr,
              'gamma': gamma,
              'step_size': step_size}

    # execute
    naive_lr = base_lr * (1 / (1 + np.exp(-gamma * (np.arange(max_iter) - step_size))))
    lr_op = lr_policies[name]['obj'](params)(iter_buf)
    with ExecutorFactory() as ex:
        compute_lr = ex.executor(lr_op, iter_buf)
        ng_lr = [compute_lr(i).item(0) for i in range(max_iter)]

        # compare
        ng.testing.assert_allclose(ng_lr, naive_lr, atol=1e-4, rtol=1e-3)
