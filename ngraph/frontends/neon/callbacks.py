# ----------------------------------------------------------------------------
# Copyright 2014-2016 Nervana Systems Inc.
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
from __future__ import division, print_function
from builtins import zip
import h5py
import logging
import os
import time
from timeit import default_timer
import weakref
from tqdm import tqdm

logger = logging.getLogger(__name__)


class Callbacks(object):
    def __init__(self, model, output_file, interval_freq, show_progress=True):
        '''
        just store a list of callbacks
        '''
        self.callbacks = list()
        if output_file is None:
            if hasattr(self, 'callback_data'):
                del self.callback_data
            # self.name sould give a unique filename
            self.callback_data = h5py.File(self.name, driver='core', backing_store=False)
        else:
            if os.path.isfile(output_file):
                logger.warn("Overwriting output file %s", output_file)
                os.remove(output_file)
            self.callback_data = h5py.File(output_file, "w")

        self.model = weakref.ref(model)

        self.add_callback(RunTimerCallback())
        self.add_callback(TrainCostCallback())
        if show_progress:
            self.add_callback(ProgressCallback(minibatch_freq=1, interval_freq=interval_freq))
        self.add_callback(TrainLoggerCallback(minibatch_freq=1, interval_freq=interval_freq))

    def __del__(self):
        try:
            self.callback_data.close()
        except Exception:
            pass

    def add_callback(self, callback, insert_pos=None):
        """
        Add a user supplied callback. Since callbacks are run serially and share data,
        order can matter.  If the default behavior (to append the callback) is not
        sufficient, insert position can be controlled.

        Arguments:
            callback (Callback): callback object to be registered
            insert_pos (int, optional): position in the list to insert the callback.
                                        Defaults to None, meaning append
        """
        if insert_pos is None:
            self.callbacks.append(callback)
        else:
            self.callbacks.insert(insert_pos, callback)

    def on_train_begin(self, iterations):
        """
        Call all registered callbacks' on_train_begin functions.

        Arguments:
            iterations (int): Total iterations
        """
        # data iterator wraps around to avoid partial minibatches
        # callbacks producing per-minibatch data need a way to preallocate
        # buffers
        config = self.callback_data.create_group('config')
        config.attrs['total_iterations'] = iterations

        for c in self.callbacks:
            c.on_train_begin(self.callback_data, self.model())

    def on_train_end(self):
        """
        Call all registered callbacks' on_train_end functions.
        """
        for c in self.callbacks:
            c.on_train_end(self.callback_data, self.model())

        self.callback_data.close()

    def on_minibatch_begin(self, minibatch):
        """
        Call all registered callbacks' on_minibatch_begin functions.

        Arguments:
            minibatch (int): index of minibatch that is beginning
        """
        for c in self.callbacks:
            if c.should_fire(minibatch, c.minibatch_freq):
                c.on_minibatch_begin(self.callback_data, self.model(), minibatch)
            if c.should_fire(minibatch, c.interval_freq):
                c.on_interval_begin(self.callback_data, self.model(), minibatch)

    def on_minibatch_end(self, minibatch):
        """
        Call all registered callbacks' on_minibatch_end functions.

        Arguments:
            minibatch (int): index of minibatch that is ending
        """
        for c in self.callbacks:
            if c.should_fire(minibatch, c.minibatch_freq):
                c.on_minibatch_end(self.callback_data, self.model(), minibatch)
            if c.should_fire(minibatch, c.interval_freq):
                c.on_interval_end(self.callback_data, self.model(), minibatch)


class Callback(object):
    """
    Interface defining common callback functions.

    Implement a callback by subclassing Callback and overriding the necessary
    on_[train,epoch,minibatch]_[begin,end] functions.

    Callback functions provide time queues as arguments but derived callback
    classes must manage their own state
    """

    def __init__(self, interval_freq, minibatch_freq):
        self.interval_freq = interval_freq
        self.minibatch_freq = minibatch_freq

    def on_train_begin(self, callback_data, model):
        """
        Called when training is about to begin

        Arguments:
            callback_data (HDF5 dataset): shared data between callbacks
            model (Model): model object
        """
        pass

    def on_train_end(self, callback_data, model):
        """
        Called when training is about to end

        Arguments:
            callback_data (HDF5 dataset): shared data between callbacks
            model (Model): model object
        """
        pass

    def on_interval_begin(self, callback_data, model, iteration_idx):
        """
        Called when an iteration interval is about to begin

        Arguments:
            callback_data (HDF5 dataset): shared data between callbacks
            model (Model): model object
            iteration_idx (int): index of iteration that is beginning
        """
        pass

    def on_interval_end(self, callback_data, model, iteration_idx):
        """
        Called when an iteration_idx is about to end

        Arguments:
            callback_data (HDF5 dataset): shared data between callbacks
            model (Model): model object
            iteration_idx (int): index of iteration that is ending
        """
        pass

    def on_minibatch_begin(self, callback_data, model, iteration_idx):
        """
        Called when a minibatch is about to begin

        Arguments:
            callback_data (HDF5 dataset): shared data between callbacks
            model (Model): model object
            epoch (int): index of current epoch
            minibatch (int): index of minibatch that is beginning
        """
        pass

    def on_minibatch_end(self, callback_data, model, iteration_idx):
        """
        Called when a minibatch is about to end

        Arguments:
            callback_data (HDF5 dataset): shared data between callbacks
            model (Model): model object
            epoch (int): index of current epoch
            minibatch (int): index of minibatch that is ending
        """
        pass

    def should_fire(self, tm, freq):
        """
        Helper function for determining if a callback should do work at a given
        interval.

        Arguments:
            time (int): current time, in an arbitrary unit
            freq (int, list, None): firing frequency, in multiples of the unit used
                                    for time, or a list of times, or None (never fire)

        Returns:
            Boolean
        """
        if ((isinstance(freq, int) and (tm + 1) % freq == 0)
                or (isinstance(freq, list) and tm in freq)):
            return True
        return False


class ProgressCallback(Callback):
    """
    Callback shows overall progress
    """

    def __init__(self, interval_freq, minibatch_freq):
        self.interval_freq = interval_freq
        self.minibatch_freq = minibatch_freq

    def on_train_begin(self, callback_data, model):
        self.tpbar = tqdm(desc="Overall",
                          unit="minibatches",
                          ncols=80,
                          total=callback_data['config'].attrs['total_iterations'])

    def on_train_end(self, callback_data, model):
        self.tpbar.close()

    def on_minibatch_end(self, callback_data, model, iteration_idx):
        self.tpbar.update(1)


class RunTimerCallback(Callback):
    """
    Callback which tracks the total training time.
    """

    def __init__(self):
        self.interval_freq = []
        self.minibatch_freq = []

    def on_train_begin(self, callback_data, model):
        timing = callback_data.create_group("time/train")
        timing.create_dataset("start_time", (1,), dtype='float64')
        timing.create_dataset("end_time", (1,), dtype='float64')
        timing['start_time'][0] = time.time()
        timing['start_time'].attrs['units'] = 'seconds'

    def on_train_end(self, callback_data, model):
        callback_data['time/train/end_time'][0] = time.time()
        callback_data['time/train/end_time'].attrs['units'] = 'seconds'


class TrainCostCallback(Callback):
    """
    Callback for computing average training cost periodically during training.
    """

    def __init__(self):
        super(TrainCostCallback, self).__init__(minibatch_freq=1, interval_freq=[])

    def on_train_begin(self, callback_data, model):
        iterations = callback_data['config'].attrs['total_iterations']
        callback_data.create_dataset("cost/train", (iterations,))
        # clue in the data reader to use the 'minibatch' time_markers
        callback_data['cost/train'].attrs['time_markers'] = 'minibatch'

    def on_minibatch_end(self, callback_data, model, iteration_idx):
        callback_data['cost/train'][iteration_idx] = model.current_batch_cost


class LossCallback(Callback):
    """
    Callback for calculating the loss on a given dataset periodically during training.

    Arguments:
        eval_set (NervanaDataIterator): dataset to evaluate
        interval_freq (int, optional): how often (in iterations) to log info.
    """

    def __init__(self, minibatch_freq, interval_freq, interval_loss_func):
        super(LossCallback, self).__init__(minibatch_freq, interval_freq)
        self.interval_loss_func = interval_loss_func

    def on_train_begin(self, callback_data, model):
        num_points = callback_data['config'].attrs['total_iterations'] // self.interval_freq
        callback_data.create_dataset("cost/loss", (num_points,))
        callback_data.create_dataset("time/loss", (num_points,))
        callback_data["cost/loss"].attrs['time_markers'] = 'interval_freq'
        callback_data["cost/loss"].attrs['interval_freq'] = self.interval_freq

    def on_interval_end(self, callback_data, model, iteration_idx):
        start_loss = default_timer()
        interval_idx = iteration_idx // self.interval_freq
        callback_data["cost/loss"][interval_idx] = self.interval_loss_func()
        callback_data["time/loss"][interval_idx] = (default_timer() - start_loss)


class MetricCallback(Callback):
    """
    Callback for calculating a metric on a given dataset periodically during
    training.

    Arguments:
        metric_funcs (Metric): metric to evaluate
        epoch_freq (int, optional): how often (in epochs) to log info.
                                    Defaults to every 1 epoch.
    """

    def __init__(self, metric_funcs, interval_freq):
        super(MetricCallback, self).__init__(interval_freq=interval_freq)
        self.metric_funcs = [m['comp'] for m in metric_funcs]
        self.metric_names = [m['name'] for m in metric_funcs]

    def on_train_begin(self, callback_data, model):
        num_points = callback_data['config'].attrs['total_iterations'] // self.interval_freq
        callback_data.create_group("metrics")

        for met in self.metric_names:
            group_name = "metrics/%s" % met
            callback_data.create_dataset(group_name, (num_points,))
            callback_data[group_name].attrs['time_markers'] = 'interval_freq'
            callback_data[group_name].attrs['interval_freq'] = self.interval_freq

    def on_interval_end(self, callback_data, model, iteration_idx):
        interval = iteration_idx // self.interval_freq

        for name, func in zip(self.metric_names, self.metric_funcs):
            callback_data["metrics/%s" % name][interval] = func()


class TrainLoggerCallback(Callback):
    """
    Callback for logging training progress.

    Arguments:
        epoch_freq (int, optional): how often (in epochs) to log training info.
                                    Defaults to every 1 epoch.
        minibatch_freq (int, optional): how often (in minibatches) to log
                                        training info, or None to log only on
                                        epoch boundaries.  Defaults to None.
    """
    def on_interval_end(self, callback_data, model, iteration_idx):
        interval = slice(iteration_idx + 1 - self.interval_freq, iteration_idx)
        train_cost = callback_data["cost/train"][interval].mean()
        tqdm.write("Iteration {} -- Avg Train cost: {}".format(iteration_idx + 1, train_cost))
        # logger.warn("Interval %d Minibatch %d complete. Train cost: %f",
        #             interval, (iteration_idx % self.interval_freq), train_cost)
