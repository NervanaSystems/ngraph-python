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
# ----------------------------------------------------------------------------\
import os
import subprocess
import h5py
import numpy as np
import ngraph as ng
import pytest

base_dir = os.path.dirname(__file__)
example_dir = os.path.join(base_dir, '../../../../examples/')
work_dir = os.getenv('NGRAPH_WORK_DIR', os.path.join(os.path.expanduser('~'), 'nervana'))
work_dir = os.path.realpath(work_dir)

db = {'mnist': {'filename': os.path.join(example_dir, 'mnist', 'mnist_mlp.py'),
                'arguments': '--data_dir {} --num_iterations {} --iter_interval {} -r 0'.format(
                    work_dir, 950, 475),
                'cost': 0.28437907},
      'cifar10': {'filename': os.path.join(example_dir, 'cifar10', 'cifar10_mlp.py'),
                  'arguments': '--data_dir {} --num_iterations {} --iter_interval {} -r 0'.format(
                  work_dir, 950, 475),
                  'cost': 1.6199253}
      }


def get_cost(filename):
    with h5py.File(filename, 'r') as f:
        cost = np.array(f['cost']['train'])
        return cost


def get_last_interval_cost(filename, interval_size):
    cost = get_cost(filename)
    return cost[-interval_size:].mean()


@pytest.fixture(scope='module', params=db.keys())
def run_model(request, tmpdir_factory, transformer_factory):
    model = request.param

    ofiles, rcs = [], []

    for i in range(2):
        ofile = tmpdir_factory.mktemp(model).join("out{}.hdf5".format(i)).__str__()
        cmd = '{} {} --out {}'.format(db[model]['filename'], db[model]['arguments'], ofile)
        rc = subprocess.check_call(cmd, shell=True)
        ofiles.append(ofile)
        rcs.append(rc)

    assert all([result == 0 for result in rcs])
    return (model, ofiles)


def test_model_traincost(run_model, transformer_factory):
    model_name, (out_path, _) = run_model
    cost = get_last_interval_cost(out_path, 10)
    print(cost)
    ng.testing.assert_allclose(cost, db[model_name]['cost'], rtol=0.15)


def test_model_positivecost(run_model, transformer_factory):
    """
    When running MNIST or cifar10 in neon, all per-minibatch costs are positive.
    """
    model_name, (out_path, _) = run_model

    cost = get_cost(out_path)
    assert all(cost >= 0.0)


def test_model_simple(run_model, transformer_factory):
    # simple test that the model runs without error
    model_name, (out_path1, out_path2) = run_model
    assert os.path.exists(out_path1)
    assert os.path.exists(out_path2)


def test_model_determinism(run_model, transformer_factory):
    model_name, (out_path1, out_path2) = run_model
    cost1 = get_last_interval_cost(out_path1, 10)
    cost2 = get_last_interval_cost(out_path2, 10)
    ng.testing.assert_allclose(cost1, cost2, rtol=1e-5)
