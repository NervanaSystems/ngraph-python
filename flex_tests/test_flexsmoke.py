from __future__ import print_function
import pytest
import os
import subprocess
import re

pytestmark = pytest.mark.transformer_dependent("module")

# known issues:
bug_1295 = pytest.mark.xfail(strict=True, reason="GitHub issue #1295: Flexgpu transformer broken in current master, "
                                                 "the lack of the MNIST convergence")


@pytest.mark.parametrize("script_path, description", (
        bug_1295(("examples/mnist/mnist_mlp.py", "Check if MNIST MLP is trainable")),
        bug_1295(("examples/cifar10/cifar10_mlp.py", "Check if CIFAR10 MLP is trainable"))
))
def test_if_mlp_network_is_trainable(script_path, description):
    """
    :param script_path: Path to the script with training of MLP neural network using specific data set
    :param description: Description what is train
    :return: If after 200 iteration misclass_pct is less than 0.1 - test case will pass, if the script returns 
            status different than 0 (some error is occurred) or misclass_pct is not less than 0.1 -  test case will fail
    Each test case tests only trainings using flex. The tests are not functional, this is only smoke test.
    """

    print("Description of test case: ", description)
    mlp_path = os.path.dirname(__file__) + "/../" + script_path

    # start MLP training for flexgpu with 200 iterations and random number generator seed equals 1
    cmd = "python " + mlp_path + " -b flexgpu -t 200 -r 1"

    output = subprocess.check_output(cmd, shell=True)
    misclass_pct = re.findall(r"misclass_pct': [-+]?\d*\.\d+", output)[-1].split()[-1]
    print("Missclassification percentage after 200 iterations: ", misclass_pct)
    assert float(misclass_pct) < 0.1
