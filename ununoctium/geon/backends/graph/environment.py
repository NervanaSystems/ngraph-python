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
from builtins import object
from contextlib import contextmanager
import threading

# TODO:
# Store default values directly with the keys, i.e. a default axis length is stored in the axis.
# When acessing values from the environment, pass the default value.  This lets us get rid of the
# environment that is used to hold default values, and some of the @with_environment uses.  Can
# have an Empty object if we need a way to throw on undefined values

# store a "batch" attribute on an axis in the environment, so that a @with_batch_axes
# can automatcially tag those axes as batch.

# replace the special-cases in the environment with a kind argument to access/set so we don't
# need all these special-purpose methods

__thread_state = threading.local()


def get_thread_state():
    return __thread_state


get_thread_state().name_scope = [None]


def get_thread_name_scope():
    return get_thread_state().name_scope


get_thread_state().ops = [None]


def get_thread_ops():
    return get_thread_state().ops


def get_current_ops():
    return get_thread_ops()[-1]


@contextmanager
def captured_ops(ops=None):
    try:
        get_thread_ops().append(ops)
        yield (ops)
    finally:
        get_thread_ops().pop()


get_thread_state().environment = [None]


def get_thread_environment():
    return get_thread_state().environment


def get_current_environment():
    return get_thread_environment()[-1]


class EnvironmentProxy(object):

    def __init__(self):
        pass

    def __getattr__(self, item):
        return get_current_environment()[item]


proxy = EnvironmentProxy()


@contextmanager
def bound_environment(environment=None, create=True):
    if environment is None and create:
        environment = Environment(parent=get_current_environment())

    try:
        get_thread_environment().append(environment)
        yield (environment)
    finally:
        get_thread_environment().pop()


class Environment(object):

    def __init__(self, parent=None, **kargs):
        super(Environment, self).__init__(**kargs)
        self.parent = parent
        self.values = dict()

    def _chained_search(self, attr, key, default=None, use_default=False):
        env = self
        while True:
            try:
                return env.__getattribute__(attr)[key]
            except KeyError:
                env = env.parent
                if env is None:
                    if use_default:
                        return default
                    raise

    def __getitem__(self, key):
        return self._chained_search('values', key)

    def __setitem__(self, key, value):
        self.values[key] = value

    def get_value(self, key, default):
        return self._chained_search(
            'values', key, default=default, use_default=True)

    def set_value(self, key, value):
        self.values[key] = value
