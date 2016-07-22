from __future__ import print_function

import os
import numpy as np
from subprocess import Popen, PIPE

'''
Test the usage of transformer.allreduce

'''
from neon.initializers import Constant
import geon.backends.graph.funs as be
import geon.backends.graph.axis as ax
from geon.backends.graph.mpihandle import MPIHandle


def test_mpi_allreduce():
    with be.bound_environment():
        ax.Y.length = 1
        hello = be.Variable(axes=(ax.Y,), init=Constant(4))

        a = np.array([[4, 1, 2, -3, 4],
                      [5, -6, 7, -8, 9]], dtype=np.float32)

        handle = MPIHandle()
        result = handle.allreduceAvg(a)

        assert(np.array_equal(a, result))

        print("pass mpi allreduce test")


parent_info = os.popen("ps -p %d" % os.getppid()).read().strip().split('\n')
parent_cmd = (parent_info[-1].split())[-1]

if parent_cmd == 'mpirun':
    test_mpi_allreduce()

else:
    cmd = "mpirun -n 2 python /home/yashar/gitlab/graphiti/ununoctium/tests/test_allreduce.py"
    proc = Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True)
    proc.wait()
    proc_out, proc_err = proc.communicate()
