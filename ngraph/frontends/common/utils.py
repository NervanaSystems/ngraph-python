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

import ngraph as ng


class CommonSGDOptimizer(object):
    def __init__(self, lrate=0.1):
        self.lrate = lrate

    def minimize(self, cost, variables):
        """
        Minimize cost by returning update Ops.

        Arguments:
            cost: The cost Op to be minimized
            variables: TODO

        Returns:
            A doall op containing setitems to variable ops.
        """

        assert cost is not None
        assert variables is not None

        grads = [ng.deriv(cost, variable) for variable in variables]
        with ng.Op.saved_user_deps():
            param_updates = [
                ng.assign(variable, variable - self.lrate * grad)
                for variable, grad in zip(variables, grads)
            ]
            updates = ng.doall(param_updates)
        return updates
