import os
import subprocess
import h5py
import numpy as np
import pytest

base_dir = os.path.dirname(__file__)
example_dir = os.path.join(base_dir, '../../../../examples/')

db = {'mnist': {'filename': os.path.join(example_dir, 'mnist_mlp.py'),
                'arguments': '-e10 -r0',
                'cost': 0.0839},
      'cifar10': {'filename': os.path.join(example_dir, 'cifar10_mlp.py'),
                  'arguments': '-e10 -r0',
                  'cost': 2.25}
      }


def get_last_epoch_cost(filename):
    with h5py.File(filename, 'r') as f:
        mb = np.array(f['time_markers']['minibatch'])
        cost = np.array(f['cost']['train'])

        tstart = int(mb[-2])
        tend = int(mb[-1] - 1)
        return cost[tstart:tend].mean()


def get_cost(filename):
    with h5py.File(filename, 'r') as f:
        mb = np.array(f['time_markers']['minibatch'])
        cost = np.array(f['cost']['train'])

        tstart = 0
        tend = int(mb[-1] - 1)
        return cost[tstart:tend]


@pytest.fixture(scope='module', params=db.keys())
def run_model(request, tmpdir_factory):
    model_name = request.param
    out1 = tmpdir_factory.mktemp(model_name).join("out1.hdf5").__str__()
    out2 = tmpdir_factory.mktemp(model_name).join("out2.hdf5").__str__()

    cmd1 = ' '.join([db[model_name]['filename'],
                     db[model_name]['arguments'],
                     '-o{}'.format(out1)])

    cmd2 = ' '.join([db[model_name]['filename'],
                     db[model_name]['arguments'],
                     '-o{}'.format(out2)])

    rc1 = subprocess.check_call(cmd1, shell=True)
    rc2 = subprocess.check_call(cmd2, shell=True)

    assert rc1 == 0 and rc2 == 0
    return (model_name, out1, out2)


@pytest.mark.xfail
def test_model_traincost(run_model):
    (model_name, out_path, _) = run_model
    cost = get_last_epoch_cost(out_path)
    np.testing.assert_allclose(cost, db[model_name]['cost'], rtol=0.15)


def test_model_positivecost(run_model):
    """
    When running MNIST or cifar10 in neon, all per-minibatch costs are positive.
    """
    (model_name, out_path, _) = run_model

    if model_name in ['mnist']:
        pytest.xfail("negative costs still an error")
    else:
        cost = get_cost(out_path)
        assert all(cost >= 0.0)


def test_model_simple(run_model):
    # simple test that the model runs without error
    (model_name, out_path1, out_path2) = run_model
    assert os.path.exists(out_path1)
    assert os.path.exists(out_path2)


def test_model_determinism(run_model):
    (model_name, out_path1, out_path2) = run_model
    cost1 = get_last_epoch_cost(out_path1)
    cost2 = get_last_epoch_cost(out_path2)
    np.testing.assert_allclose(cost1, cost2, rtol=1e-5)
