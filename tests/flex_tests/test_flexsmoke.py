from __future__ import print_function
import pytest
import os
import subprocess
import re

pytestmark = [pytest.mark.transformer_dependent, pytest.mark.flex_only]


@pytest.mark.parametrize("script_path, description, misclass_threshold", (
    ("examples/mnist/mnist_mlp.py", "Check if MNIST MLP is trainable", 0.1),
    ("examples/cifar10/cifar10_mlp.py", "Check if CIFAR10 MLP is trainable", 0.65)
))
def test_if_mlp_network_is_trainable(script_path, description, misclass_threshold):
    """
    :param script_path: Path to the script with training of MLP neural network using specific data
    set
    :param description: Description what is train
    :param misclass_threshold Accepted missclassification threshold after 200 iterations
    :return: If after 200 iteration misclass_pct is less than 0.1 - test case will pass, if the
    script returns status different than 0 (some error is occurred) or misclass_pct is not less
    than 0.1 test case will fail
    Each test case tests only trainings using flex. The tests are not functional, this is only
    smoke test.
    """

    print("Description of test case: ", description)
    mlp_path = os.path.dirname(__file__) + "/../../" + script_path

    # start MLP training for flexgpu with 200 iterations and random number generator seed equals 1
    cmd = "python " + mlp_path + " -b flexgpu -t 200 -r 1"

    output = subprocess.check_output(cmd, shell=True).decode("utf-8")
    misclass_pct = re.findall(r"misclass_pct': [-+]?\d*\.\d+", output)[-1].split()[-1]
    print("Missclassification percentage after 200 iterations: ", misclass_pct)
    assert float(misclass_pct) < misclass_threshold
