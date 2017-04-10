from __future__ import print_function
import pytest
import os
import subprocess

pytestmark = pytest.mark.transformer_dependent("module")


@pytest.mark.parametrize("script_path, description", (
    ("examples/mnist/mnist_mlp.py", "Check if MNIST MLP is trainable"),
    ("examples/cifar10/cifar10_mlp.py", "Check if CIFAR10 MLP is trainable")
))
def test_if_network_is_trainable(script_path, description):
    """
    :param script_path: Path to the script with training of selected neural network using specific data set
    :param description: Description what is train
    :return: If selected script returns 0 (training finished successfully) test case will pass, if the script returns 
            status different than 0, test case will fail (it means that script finished abnormally).
    Each test case tests only trainings using flex. The tests are not functional so it's not important what values are
    returned.
    """
    print ("Description of test case: ", description)
    mnist_mlp_path = os.path.dirname(__file__) + "/../" + script_path
    cmd = "python " + mnist_mlp_path + " -b flexgpu"
    subprocess.check_output(cmd, shell=True)
