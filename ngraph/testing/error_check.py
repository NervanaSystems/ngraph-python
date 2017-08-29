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
# import pytest
import numpy as np
import ngraph as ng


def transformer_name():
    return ng.transformers.base.__transformer_factory.name


def __overwrite_rtol_atol(rtol, atol, desired):
    """
    Overwrite atol, rtol by the transformer's default atol rtol if the default
    is less strict.

    Args:
        rtol: float, relative tolerance
        atol: float, absolute tolerance

    Returns:
        The updated atol, rtol
    """
    try:
        # get transformer name
        name = transformer_name()
        # get transformer class
        tr = ng.transformers.Transformer.transformers[name]
        # get default atol, rtol according to used transformer
        tr.default_atol, tr.default_rtol = tr.get_default_tolerance(desired)

        # rewrite rtol, atol if default is coarser
        rtol = tr.default_rtol if tr.default_rtol > rtol else rtol
        atol = tr.default_atol if tr.default_atol > atol else atol
    except:
        # if transformer not found, not make changes
        pass
    return rtol, atol


def assert_allclose(actual, desired, rtol=1e-07, atol=0, equal_nan=False,
                    err_msg='', verbose=True, transformer_overwrite=True):
    """
    Pass through for numpy.testing.assert_allclose, with support for rtol and
    atol overwrite for different transformers.

    Args:
        actual: array_like, actual value
        desired: array_like, desired value
        rtol: float, relative tolerance
        atol: float, relative tolerance
        equal_nan: bool, whether to compare NaNs as equal
        err_msg: str, error message to be printed in case of failure.
        verbose: bool, if True, the conflicting values are appended to the error message
        transofrmer_overwrite: when True, use transformer's atol / rtol if they
                               are less strict than atol / rtol

    TODO: handle heter
    """
    if transformer_overwrite:
        rtol, atol = __overwrite_rtol_atol(rtol, atol, desired)
    np.testing.assert_allclose(actual, desired, rtol=rtol, atol=atol,
                               equal_nan=equal_nan, err_msg=err_msg,
                               verbose=verbose)
