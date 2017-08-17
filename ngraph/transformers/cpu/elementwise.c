/*******************************************************************************
* Copyright 2016 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/
#include "mkldnn_engine.h"
#include "mkldnn_util.h"

/* Create list of mkldnn primitives to run elelment Wise add primitive  */
void create_mkldnn_add_kernel(mkldnn_engine_t engine, int src1_dims,
                              int src2_dims, int dst_dims, int* src1_sizes,
                              int* src2_sizes, int* dst_sizes,
                              mkldnn_memory_desc_t* src1_md,
                              mkldnn_memory_desc_t* src2_md,
                              int num_matrix_to_add,
                              mkldnn_data_type_t data_type,
                              mkldnn_opkernel_t opkernel) {
  assert(src1_dims == 1);
  assert(src2_dims == 1);
  assert(dst_dims == 1);
  assert(num_matrix_to_add == 2);

  // create memory primitive descriptor
  mkldnn_memory_desc_t md1;
  mkldnn_primitive_desc_t pd1;
  if (src1_md) {
    md1 = *src1_md;
  } else {
    MKL_CHECK(mkldnn_memory_desc_init(&md1, src1_dims, src1_sizes, data_type,
                                      mkldnn_x));
  }
  MKL_CHECK(mkldnn_memory_primitive_desc_create(&pd1, &md1, engine));

  mkldnn_memory_desc_t md2;
  mkldnn_primitive_desc_t pd2;
  if (src2_md) {
    md2 = *src2_md;
  } else {
    MKL_CHECK(mkldnn_memory_desc_init(&md2, src2_dims, src2_sizes, data_type,
                                      mkldnn_x));
  }
  MKL_CHECK(mkldnn_memory_primitive_desc_create(&pd2, &md2, engine));

  const_mkldnn_primitive_desc_t input_pds[] = {pd1, pd2};

  // create a Sum primitive descriptor
  double scale_vector[] = {1, 1};
  MKL_CHECK(mkldnn_sum_primitive_desc_create(
      &opkernel->op_desc, NULL, num_matrix_to_add, scale_vector, input_pds));

  // create a memory primitive for input and output
  if (src1_md) {
    create_mkldnn_tensor_from_md(src1_dims, src1_sizes, src1_md, engine,
                                 &(opkernel->inputs[0]));
  } else {
    create_mkldnn_tensor(src1_dims, src1_sizes, data_type, mkldnn_x, engine,
                         &(opkernel->inputs[0]));
  }

  if (src2_md) {
    create_mkldnn_tensor_from_md(src2_dims, src2_sizes, src2_md, engine,
                                 &(opkernel->inputs[1]));
  } else {
    create_mkldnn_tensor(src2_dims, src2_sizes, data_type, mkldnn_x, engine,
                         &(opkernel->inputs[1]));
  }
  mkldnn_memory_desc_t dst_md = md1;
  create_mkldnn_tensor_from_md(dst_dims, dst_sizes, &dst_md, engine,
                               &(opkernel->outputs[0]));
  opkernel->num_inputs = 2;
  opkernel->num_outputs = 1;

  // No reorders required
  opkernel->reorder_i[0] = NULL;
  opkernel->reorder_i[1] = NULL;
  opkernel->reorder_o[0] = NULL;

  // create sum primitive
  const_mkldnn_primitive_t add_prim_dsts[] = {opkernel->outputs[0].prim};
  mkldnn_primitive_at_t add_prim_srcs[] = {
      mkldnn_primitive_at(opkernel->inputs[0].prim, 0),
      mkldnn_primitive_at(opkernel->inputs[1].prim, 0),
  };

  MKL_CHECK(mkldnn_primitive_create(&opkernel->op_prim, opkernel->op_desc,
                                    add_prim_srcs, add_prim_dsts));

  opkernel->net[opkernel->net_size++] = opkernel->op_prim;
}
