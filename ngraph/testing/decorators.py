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

import decorator
import numpy as np


def with_error_settings(**new_settings):
    """
    TODO.

    Arguments:
      **new_settings: TODO

    Returns:
    """
    @decorator.decorator
    def dec(f, *args, **kwargs):
        old_settings = np.geterr()

        np.seterr(**new_settings)
        ret = f(*args, **kwargs)

        np.seterr(**old_settings)

        return ret

    return dec


def raise_all_numpy_errors(f):
    """
    TODO.

    Arguments:
      f: TODO

    Returns:
    """
    settings = {k: 'raise' for k in ['divide', 'over', 'under', 'invalid']}
    return with_error_settings(**settings)(f)
