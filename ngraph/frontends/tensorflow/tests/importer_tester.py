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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import ngraph as ng
import os
import ngraph.transformers as ngt
import pytest
from ngraph.frontends.tensorflow.tf_importer.importer import TFImporter
import tempfile


@pytest.mark.usefixtures("transformer_factory")
class ImporterTester(object):
    """
    Tester class for py.test
    """

    @pytest.fixture(autouse=True)
    def build_transformer(self, transformer_factory):
        pass

    @classmethod
    def setup_class(self):
        self.tmp_file = tempfile.NamedTemporaryFile(suffix='.txt')
        self.pb_txt_path = self.tmp_file.name

    def setup_method(self, method):
        self.sess = tf.Session()

    def teardown_method(self, method, delete_dump=True):
        # close session - doesn't work
        self.sess.close()

        # clear sess.graph_def
        tf.reset_default_graph()

        # remove dumped protobuf
        if delete_dump:
            try:
                os.remove(self.pb_txt_path)
            except:
                print("[clean up] test dump does not exist")

    def run(self,
            tf_target_node,
            tf_init_op=None,
            tf_feed_dict=None,
            print_tf_result=False,
            print_ng_result=False,
            verbose=False,
            rtol=1e-05,
            atol=1e-08):
        """
        Performs test with optional feed_dicts, compares result of TF and ngraph
        Args:
            tf_target_node: target node in tf
            tf_init_op: init op in tf
            tf_feed_dict: feed_dict in tf
            print_tf_result: prints tf_result if set to True
            print_ng_result: prints ng_result if set to True
            verbose: prints tf's node_def if set to True
        """
        # run TF
        tf_result = self.tf_run(
            tf_target_node=tf_target_node,
            tf_init_op=tf_init_op,
            tf_feed_dict=tf_feed_dict,
            print_tf_result=print_ng_result)

        # run NG
        ng_result = self.ng_run(
            tf_target_node=tf_target_node,
            tf_init_op=tf_init_op,
            tf_feed_dict=tf_feed_dict,
            print_ng_result=print_ng_result,
            verbose=verbose)

        # assert
        assert tf_result.shape == ng_result.shape
        assert ng.testing.allclose(tf_result, ng_result, rtol=rtol, atol=atol)

    def ng_run(self,
               tf_target_node,
               tf_init_op=None,
               tf_feed_dict=None,
               print_ng_result=False,
               verbose=False):
        """
        Run and get ngrpah results
        Args:
            tf_target_node: target node in tf
            tf_feed_dict: feed_dict in tf
            print_ng_result: prints ng_result if set to True
            verbose: prints tf's node_def if set to True

        Returns:
            ng_result
        """
        # init importer, transformer
        importer = TFImporter()
        importer.import_protobuf(self.pb_txt_path, verbose=verbose)
        transformer = ngt.make_transformer()

        # set target node
        ng_target_node = importer.get_op_handle_by_name(
            tf_target_node.name[:-2])

        # init op
        ng_init_op = importer.get_op_handle(tf_init_op) if tf_init_op else None
        ng_init_comp = transformer.computation(ng_init_op)

        # evaluate ngraph
        if tf_feed_dict is not None:
            # get targeting nodes for ng, convert tf's feed dict to list
            tf_placeholder_nodes = [node for (node, _) in tf_feed_dict.items()]
            tf_placeholder_names = [node.name for node in tf_placeholder_nodes]
            ng_placeholder_nodes = [
                importer.get_op_handle_by_name(name[:-2])
                for name in tf_placeholder_names
            ]
            ng_placeholder_vals = [val for (_, val) in tf_feed_dict.items()]

            # evaluate ngraph result
            ng_result_comp = transformer.computation(ng_target_node,
                                                     *ng_placeholder_nodes)
            if ng_init_op:
                ng_init_comp()
            ng_result = ng_result_comp(*ng_placeholder_vals)
        else:
            ng_result_comp = transformer.computation(ng_target_node)
            if ng_init_op:
                ng_init_comp()
            ng_result = ng_result_comp()
        if print_ng_result:
            print(ng_result)

        transformer.close()
        return ng_result

    def tf_run(self,
               tf_target_node,
               tf_init_op=None,
               tf_feed_dict=None,
               print_tf_result=False):
        """
        Run and get tf results
        Args:
            tf_target_node: target node in tf
            tf_init_op: init op in tf
            tf_feed_dict: feed_dict in tf
            print_tf_result: prints tf_result if set to True

        Returns:
            tf_result
        """
        # init
        if tf_init_op:
            self.sess.run(tf_init_op)

        # get tensorflow result
        tf_result = self.sess.run(tf_target_node, feed_dict=tf_feed_dict)
        if print_tf_result:
            print(tf_result)

        # write to protobuf
        tf.train.write_graph(self.sess.graph_def, "./", self.pb_txt_path, True)

        return tf_result
