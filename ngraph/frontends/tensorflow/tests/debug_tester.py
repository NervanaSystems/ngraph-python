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
from __future__ import print_function
from ngraph.frontends.tensorflow.tests.importer_tester import ImporterTester
import tensorflow as tf
import os
import re
import time
import atexit
import ngraph as ng

cmd_kill = 'pid=`lsof -t -i:6006`; if [ $pid ] ; then kill -9 $pid; fi'
cmd_browser = 'open http://0.0.0.0:6006/#graphs'
cmd_start_tensorboard = 'tensorboard --logdir=.'


def remove_event_dump(dir='./', pattern='events.out.tfevents.*'):
    for f in os.listdir(dir):
        if re.search(pattern, f):
            target_file = os.path.join(dir, f)
            os.remove(target_file)
            print("Removed event dump: %s" % target_file)


def clean_up():
    os.system(cmd_kill)
    remove_event_dump()


def start_tensorboard():
    """
    Start tensorboard on current directory
    """
    os.system(cmd_kill)
    os.system(cmd_browser)
    os.system(cmd_start_tensorboard)


class DebugTester(ImporterTester):
    """
    Run tester directly for debugging without py.test
    """

    def __init__(self):
        pass

    def setup(self):
        remove_event_dump()
        self.setup_class()
        self.setup_method(None)

    def run(self,
            tf_target_node,
            tf_init_op=None,
            tf_feed_dict=None,
            print_tf_result=True,
            print_ng_result=True,
            verbose=False,
            rtol=1e-05,
            atol=1e-08):
        super(DebugTester, self).run(tf_target_node=tf_target_node,
                                     tf_init_op=tf_init_op,
                                     tf_feed_dict=tf_feed_dict,
                                     print_tf_result=print_ng_result,
                                     verbose=verbose,
                                     rtol=rtol,
                                     atol=atol)
        # dump graph for tensorboard
        tf.train.SummaryWriter('./', self.sess.graph)

    def tf_run(self,
               tf_target_node,
               tf_init_op=None,
               tf_feed_dict=None,
               print_tf_result=True):
        # dump graph for tensorboard
        tf.train.SummaryWriter('./', self.sess.graph)
        return super(DebugTester, self).tf_run(
            tf_target_node=tf_target_node,
            tf_init_op=tf_init_op,
            tf_feed_dict=tf_feed_dict,
            print_tf_result=print_tf_result)

    def teardown(self, delete_dump=False):
        self.teardown_method(None, delete_dump=delete_dump)
        if delete_dump:
            time.sleep(0.5)
            remove_event_dump()


def def_target_feed_dict():
    """
    Define computation and feed dict here
    """

    a = tf.ones((2, 3))
    b = tf.ones((2, 1))
    c = tf.add(a, b)

    result = c
    feed_dict = {}
    init_op = None

    # return
    return result, feed_dict, init_op


if __name__ == '__main__':
    # remove event dump at exit
    atexit.register(clean_up)

    # init
    tester = DebugTester()
    tester.setup()

    # get target node and feed_dict
    target, feed_dict, init_op = def_target_feed_dict()

    # run & teardown
    tf_results = tester.tf_run(
        target,
        tf_init_op=init_op,
        tf_feed_dict=feed_dict,
        print_tf_result=False)
    ng_results = tester.ng_run(
        target, tf_feed_dict=feed_dict, print_ng_result=False, verbose=False)
    print(tf_results)
    print(ng_results)
    assert ng.testing.allclose(tf_results, ng_results, rtol=1e-3, atol=1e-3)
    tester.teardown(delete_dump=False)

    # start tensorboard (optional)
    # start_tensorboard()
