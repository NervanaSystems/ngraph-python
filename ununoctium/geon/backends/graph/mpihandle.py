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
from __future__ import division
from builtins import object

import numpy as np

from mpi4py import MPI

comm = MPI.COMM_WORLD


class MPIHandle(object):
    MPICallDict = {
        "max": MPI.MAX,
        "min": MPI.MIN,
        "sum": MPI.SUM,
        "prod": MPI.PROD,
        "land": MPI.LAND,
        "band": MPI.LOR,
        "BOR": MPI.BOR,
        "LXOR": MPI.LXOR,
        "bxor": MPI.BXOR,
        "maxloc": MPI.MAXLOC,
        "minloc": MPI.MINLOC
    }

    def __init__(self, name="mpi_handle", **kargs):
        super(MPIHandle, self).__init__(**kargs)
        self.name = name

    def reduce(self, x, op=MPI.SUM):
        raise NotImplementedError()

    def allreduce(self, x, op="sum"):
        if not isinstance(x, np.ndarray):
            raise NotImplementedError("Input x should be a numpy ndarray")

        mpi_op = self.MPICallDict.get(op)
        recv_buffer = np.zeros(shape=x.shape, dtype=x.dtype)
        comm.Allreduce(x, recv_buffer, op=mpi_op)

        return recv_buffer

    def allreduceAvg(self, x):
        mpi_size = comm.Get_size()
        return (self.allreduce(x, op="sum") / mpi_size)

    def bcast(self, value, root=0):
        raise NotImplementedError()

    def scatter(self, value, local_value, root=0):
        raise NotImplementedError()

    def scatterv(self, recv_buff, root=0, *values):
        raise NotImplementedError()

    def gatherv(self, xlocal, root=0, *values):
        raise NotImplementedError()
